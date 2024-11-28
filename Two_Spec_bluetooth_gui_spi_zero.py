#!/usr/bin/env python3

import dbus
import dbus.exceptions
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib
import sys
import threading
import spidev
import time
import RPi.GPIO as GPIO

# Constants for the Bluetooth Service and Characteristic UUIDs
GATT_SERVICE_UUID = "d647a4b6-a14c-4695-b517-c83c1636617b"
GATT_CHARACTERISTIC_UUID = "70c4f540-3725-49da-b0bd-3f6ff779d6b5"

BLUEZ_SERVICE_NAME = 'org.bluez'
DBUS_OM_IFACE = 'org.freedesktop.DBus.ObjectManager'
LE_ADVERTISING_MANAGER_IFACE = 'org.bluez.LEAdvertisingManager1'
GATT_MANAGER_IFACE = 'org.bluez.GattManager1'
AD_PROPERTIES_IFACE = 'org.freedesktop.DBus.Properties'
GATT_SERVICE_IFACE = 'org.bluez.GattService1'
GATT_CHARACTERISTIC_IFACE = 'org.bluez.GattCharacteristic1'

# Define spectrometer configurations
spectrometers = [
    {
        'name': 'Spectrometer1',
        'spi_bus': 0,
        'cs0': 0,
        'cs1': 1,
        'data_ready_pin': 24,  # GPIO pin for Spectrometer 1
        'id': 1
    },
    {
        'name': 'Spectrometer2',
        'spi_bus': 1,
        'cs0': 0,
        'cs1': 1,
        'data_ready_pin': 23,  # GPIO pin for Spectrometer 2
        'id': 2
    }
]

# GPIO setup for both spectrometers
GPIO.setmode(GPIO.BCM)
for spectrometer in spectrometers:
    GPIO.setup(spectrometer['data_ready_pin'], GPIO.IN)

# SPI setup
SPI_MAX_SPEED_HZ = 25000000  # 25 MHz
SPI_MODE = 0b01  # CPOL=0, CPHA=1

# Register addresses
SENSOR_CTRL_ADDRESS = 8
SENSOR_EXP_TIME_LSB_ADDRESS = 9
SENSOR_EXP_TIME_MSB_ADDRESS = 10
TEMPERATURE_REGISTER_ADDRESS = 11

bus = None
mainloop = None

def find_adapter(bus):
    """Find the Bluetooth adapter."""
    remote_om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, "/"), DBUS_OM_IFACE)
    objects = remote_om.GetManagedObjects()
    for path, interfaces in objects.items():
        if LE_ADVERTISING_MANAGER_IFACE in interfaces and GATT_MANAGER_IFACE in interfaces:
            return path
    return None

class Application(dbus.service.Object):
    """BLE Application."""
    def __init__(self, bus, spi):
        self.path = '/'
        self.services = []
        dbus.service.Object.__init__(self, bus, self.path)
        self.add_service(TestService(bus, 0, spi))
        print("Application initialized, service added")

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_service(self, service):
        self.services.append(service)

    def get_services(self):
        return self.services

    @dbus.service.method(DBUS_OM_IFACE, out_signature='a{oa{sa{sv}}}')
    def GetManagedObjects(self):
        response = {}
        for service in self.services:
            response[service.get_path()] = service.get_properties()
            chrcs = service.get_characteristics()
            for chrc in chrcs:
                response[chrc.get_path()] = chrc.get_properties()
        return response

class TestService(dbus.service.Object):
    """BLE Service."""
    PATH_BASE = '/org/bluez/example/service'

    def __init__(self, bus, index, spi):
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.uuid = GATT_SERVICE_UUID
        self.primary = True
        self.characteristics = []
        dbus.service.Object.__init__(self, bus, self.path)

        # Add characteristic
        self.add_characteristic(TestCharacteristic(bus, 0, self, spi))
        print(f'Service {GATT_SERVICE_UUID} registered')

    def get_properties(self):
        return {
            GATT_SERVICE_IFACE: {
                'UUID': self.uuid,
                'Primary': self.primary,
                'Characteristics': dbus.Array(self.get_characteristic_paths(), signature='o'),
            }
        }

    def get_path(self):
        """Return the path of the service."""
        return dbus.ObjectPath(self.path)

    def get_characteristic_paths(self):
        return [chrc.get_path() for chrc in self.characteristics]

    def add_characteristic(self, characteristic):
        self.characteristics.append(characteristic)

    def get_characteristics(self):
        return self.characteristics

class TestCharacteristic(dbus.service.Object):
    """BLE Characteristic."""
    def __init__(self, bus, index, service, spi):
        self.path = service.get_path() + '/char' + str(index)
        self.bus = bus
        self.uuid = GATT_CHARACTERISTIC_UUID
        self.service = service
        self.flags = ['read', 'write', 'notify']
        self.value = []
        self.notifying = False
        self.spi = spi
        self.reading_thread = None
        self.reading_thread_event = threading.Event()
        self.spi_lock = threading.Lock()
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            GATT_CHARACTERISTIC_IFACE: {
                'Service': self.service.get_path(),
                'UUID': self.uuid,
                'Flags': self.flags,
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method(GATT_CHARACTERISTIC_IFACE, in_signature='a{sv}', out_signature='ay')
    def ReadValue(self, options):
        print('Read request received')
        return dbus.Array(self.value, signature='y')

    @dbus.service.method(GATT_CHARACTERISTIC_IFACE, in_signature='aya{sv}')
    def WriteValue(self, value, options):
        """Handle Bluetooth Write requests."""
        # Decode message to a string
        message = ''.join([chr(byte) for byte in value])
        print(f'Received message via Bluetooth: "{message}"')
        if message == 'start':
            print('Starting NIRS analysis...')
            self.start_reading()
        elif message == 'stop':
            print('Stopping NIRS analysis...')
            self.stop_reading()
        elif message.startswith('integration_time:'):
            try:
                # Assuming integration time is sent in milliseconds
                integration_time_ms = float(message.split(':')[1])
                integration_time_units = int(integration_time_ms * 1e6 / 200)  # Convert ms to units of 200 ns
                if integration_time_units < 6:
                    integration_time_units = 6  # Minimum value as per documentation
                lsb = integration_time_units & 0xFFFF
                msb = (integration_time_units >> 16) & 0xFFFF
                # Apply integration time to both spectrometers
                for spectrometer in spectrometers:
                    spi_bus = spectrometer['spi_bus']
                    cs0 = spectrometer['cs0']
                    with self.spi_lock:
                        self.spi.open(spi_bus, cs0)
                        self.spi.mode = SPI_MODE
                        self.spi.max_speed_hz = SPI_MAX_SPEED_HZ
                        self.write_register(SENSOR_EXP_TIME_LSB_ADDRESS, lsb)
                        self.write_register(SENSOR_EXP_TIME_MSB_ADDRESS, msb)
                        self.spi.close()
                print(f"Integration time set to {integration_time_ms} ms for both spectrometers.")
            except ValueError:
                print("Invalid integration time received")
        elif message == 'Reference':
            print('Reference spectra measurement requested...')
            threading.Thread(target=self.read_reference_data).start()

    def start_reading(self):
        if self.reading_thread is None:
            self.reading_thread_event.clear()
            self.reading_thread = threading.Thread(target=self.read_data)
            self.reading_thread.start()
            print("Data reading thread started.")

    def stop_reading(self):
        if self.reading_thread is not None:
            self.reading_thread_event.set()
            self.reading_thread.join()
            self.reading_thread = None
            print("Data reading thread stopped.")

    def read_register(self, address):
        """Read a 16-bit value from the given register address."""
        cmd = ((address & 0x3F) << 2) | 0x02  # Set read bit (bit 1)
        response = self.spi.xfer2([cmd, 0x00, 0x00])
        data_high = response[1]
        data_low = response[2]
        value = (data_high << 8) | data_low
        return value

    def write_register(self, address, value):
        """Write a 16-bit value to the given register address."""
        cmd = (address & 0x3F) << 2  # Clear read bit
        data_high = (value >> 8) & 0xFF
        data_low = value & 0xFF
        self.spi.xfer2([cmd, data_high, data_low])

    def read_data(self):
        NUM_PIXELS = 256  # Update this according to your detector
        BYTES_PER_PIXEL = 2
        while not self.reading_thread_event.is_set():
            for spectrometer in spectrometers:
                spi_bus = spectrometer['spi_bus']
                cs0 = spectrometer['cs0']
                cs1 = spectrometer['cs1']
                data_ready_pin = spectrometer['data_ready_pin']
                spectrometer_id = spectrometer['id']

                # Start an exposure
                with self.spi_lock:
                    self.spi.open(spi_bus, cs0)
                    self.spi.mode = SPI_MODE
                    self.spi.max_speed_hz = SPI_MAX_SPEED_HZ
                    # Start exposure by writing 1 to bit 0 of SENSOR_CTRL register
                    self.write_register(SENSOR_CTRL_ADDRESS, 0x0001)
                    self.spi.close()

                # Wait for DATA_READY_PIN to go HIGH
                start_time = time.time()
                TIMEOUT_SECONDS = 5
                while GPIO.input(data_ready_pin) != GPIO.HIGH:
                    if time.time() - start_time > TIMEOUT_SECONDS:
                        print(f"Timeout waiting for data ready signal on {spectrometer['name']}.")
                        # Abort exposure and perform soft reset
                        with self.spi_lock:
                            self.spi.open(spi_bus, cs0)
                            self.spi.mode = SPI_MODE
                            self.spi.max_speed_hz = SPI_MAX_SPEED_HZ
                            self.write_register(SENSOR_CTRL_ADDRESS, 0x0018)
                            self.spi.close()
                        continue  # Move to the next spectrometer
                    if self.reading_thread_event.is_set():
                        return
                    time.sleep(0.01)

                with self.spi_lock:
                    # Read spectral data from CS1
                    self.spi.open(spi_bus, cs1)
                    self.spi.mode = SPI_MODE
                    self.spi.max_speed_hz = SPI_MAX_SPEED_HZ
                    data = self.spi.readbytes(NUM_PIXELS * BYTES_PER_PIXEL)
                    self.spi.close()

                    # Read temperature from CS0
                    self.spi.open(spi_bus, cs0)
                    self.spi.mode = SPI_MODE
                    self.spi.max_speed_hz = SPI_MAX_SPEED_HZ
                    temperature_value = self.read_register(TEMPERATURE_REGISTER_ADDRESS)
                    # Abort exposure and perform soft reset
                    self.write_register(SENSOR_CTRL_ADDRESS, 0x0018)
                    self.spi.close()

                print(f"Received data from {spectrometer['name']} via SPI.")
                print(f"Temperature value: {temperature_value}")

                # Convert temperature_value to bytes
                temperature_data = temperature_value.to_bytes(2, byteorder='big')
                # Combine spectral data and temperature data
                combined_data = bytes(data) + temperature_data
                # Include spectrometer ID in the data
                packet = {'id': spectrometer_id, 'data': combined_data}
                GLib.idle_add(self.send_notification, packet)
                # Sleep a short time between spectrometer reads if necessary
                time.sleep(0.1)

    def send_notification(self, packet):
        # Send the combined data in chunks
        PACKET_TYPE_DATA_CHUNK = 0x01
        spectrometer_id = packet['id']
        combined_data = packet['data']
        self.send_data_in_chunks(combined_data, PACKET_TYPE_DATA_CHUNK, spectrometer_id)
        return False

    def read_reference_data(self):
        for spectrometer in spectrometers:
            spi_bus = spectrometer['spi_bus']
            cs0 = spectrometer['cs0']
            cs1 = spectrometer['cs1']
            data_ready_pin = spectrometer['data_ready_pin']
            spectrometer_id = spectrometer['id']

            # Start an exposure
            with self.spi_lock:
                self.spi.open(spi_bus, cs0)
                self.spi.mode = SPI_MODE
                self.spi.max_speed_hz = SPI_MAX_SPEED_HZ
                self.write_register(SENSOR_CTRL_ADDRESS, 0x0001)
                self.spi.close()

            # Wait for DATA_READY_PIN to go HIGH
            start_time = time.time()
            TIMEOUT_SECONDS = 5
            while GPIO.input(data_ready_pin) != GPIO.HIGH:
                if time.time() - start_time > TIMEOUT_SECONDS:
                    print(f"Timeout waiting for data ready signal on {spectrometer['name']}.")
                    # Abort exposure and perform soft reset
                    with self.spi_lock:
                        self.spi.open(spi_bus, cs0)
                        self.spi.mode = SPI_MODE
                        self.spi.max_speed_hz = SPI_MAX_SPEED_HZ
                        self.write_register(SENSOR_CTRL_ADDRESS, 0x0018)
                        self.spi.close()
                    continue  # Move to the next spectrometer
                time.sleep(0.1)

            NUM_PIXELS = 256  # Update this according to your detector
            BYTES_PER_PIXEL = 2
            with self.spi_lock:
                # Read reference spectral data from CS1
                self.spi.open(spi_bus, cs1)
                self.spi.mode = SPI_MODE
                self.spi.max_speed_hz = SPI_MAX_SPEED_HZ
                data = self.spi.readbytes(NUM_PIXELS * BYTES_PER_PIXEL)
                self.spi.close()

                # Read temperature from CS0
                self.spi.open(spi_bus, cs0)
                self.spi.mode = SPI_MODE
                self.spi.max_speed_hz = SPI_MAX_SPEED_HZ
                temperature_value = self.read_register(TEMPERATURE_REGISTER_ADDRESS)
                # Abort exposure and perform soft reset
                self.write_register(SENSOR_CTRL_ADDRESS, 0x0018)
                self.spi.close()

            print(f"Received reference data from {spectrometer['name']} via SPI.")
            print(f"Temperature value: {temperature_value}")

            # Convert temperature_value to bytes
            temperature_data = temperature_value.to_bytes(2, byteorder='big')
            # Combine spectral data and temperature data
            combined_data = bytes(data) + temperature_data
            packet = {'id': spectrometer_id, 'data': combined_data}
            GLib.idle_add(self.send_reference_data, packet)
            # Sleep a short time between spectrometer reads if necessary
            time.sleep(0.1)

    def send_reference_data(self, packet):
        # Send the reference data in chunks
        PACKET_TYPE_REFERENCE_DATA = 0x03
        spectrometer_id = packet['id']
        combined_data = packet['data']
        self.send_data_in_chunks(combined_data, PACKET_TYPE_REFERENCE_DATA, spectrometer_id)
        return False

    def send_data_in_chunks(self, combined_data, packet_type, spectrometer_id):
        """Split the data into chunks and send over BLE."""
        CHUNK_SIZE = 160  # Adjust according to MTU and BLE limitations

        total_length = len(combined_data)
        total_chunks = (total_length + CHUNK_SIZE - 1) // CHUNK_SIZE

        for chunk_number in range(total_chunks):
            start_index = chunk_number * CHUNK_SIZE
            end_index = min(start_index + CHUNK_SIZE, total_length)
            data_chunk = combined_data[start_index:end_index]

            # Create header
            header = bytearray()
            header.append(packet_type)       # Packet Type
            header.append(chunk_number)      # Sequence Number
            header.append(total_chunks)      # Total Chunks
            header.append(spectrometer_id)   # Spectrometer ID

            packet = header + data_chunk

            # Send the notification
            self.send_chunk_notification(packet)

            # Wait a short time between sends if necessary
            time.sleep(0.01)  # Adjust sleep time as needed

    def send_chunk_notification(self, packet):
        if self.notifying:
            value = dbus.Array(packet, signature='y')
            print(f"Sending chunk {packet[1]+1}/{packet[2]} for spectrometer {packet[3]} with length {len(value)}")
            self.PropertiesChanged(GATT_CHARACTERISTIC_IFACE, {'Value': value}, [])

    @dbus.service.method(GATT_CHARACTERISTIC_IFACE)
    def StartNotify(self):
        if self.notifying:
            return
        self.notifying = True
        print(f"Start notifying {self.uuid}")

    @dbus.service.method(GATT_CHARACTERISTIC_IFACE)
    def StopNotify(self):
        if not self.notifying:
            return
        self.notifying = False
        print(f"Stop notifying {self.uuid}")

    @dbus.service.signal(dbus_interface='org.freedesktop.DBus.Properties', signature='sa{sv}as')
    def PropertiesChanged(self, interface, changed_properties, invalidated_properties):
        pass

class Advertisement(dbus.service.Object):
    """BLE Advertisement."""
    PATH_BASE = '/org/bluez/example/advertisement'

    def __init__(self, bus, index, advertising_type):
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.ad_type = advertising_type
        self.service_uuids = [GATT_SERVICE_UUID]
        self.local_name = 'RaspberryPi-BLE'
        self.includes = ['tx-power']
        dbus.service.Object.__init__(self, bus, self.path)
        print(f"Advertisement created for {self.local_name}")

    def get_properties(self):
        properties = {
            'Type': self.ad_type,
            'ServiceUUIDs': dbus.Array(self.service_uuids, signature='s'),
            'LocalName': dbus.String(self.local_name),
            'Includes': dbus.Array(self.includes, signature='s'),
        }
        return {'org.bluez.LEAdvertisement1': properties}

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method(AD_PROPERTIES_IFACE, in_signature='ss', out_signature='v')
    def Get(self, interface, prop):
        if interface != 'org.bluez.LEAdvertisement1':
            raise dbus.exceptions.DBusException(
                'org.freedesktop.DBus.Error.InvalidArgs',
                f'Invalid interface {interface}'
            )
        return self.get_properties()['org.bluez.LEAdvertisement1'][prop]

    @dbus.service.method(AD_PROPERTIES_IFACE, in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != 'org.bluez.LEAdvertisement1':
            raise dbus.exceptions.DBusException(
                'org.freedesktop.DBus.Error.InvalidArgs',
                f'Invalid interface {interface}'
            )
        return self.get_properties()['org.bluez.LEAdvertisement1']

    @dbus.service.method('org.bluez.LEAdvertisement1')
    def Release(self):
        print(f'{self.path}: Released!')

def main():
    global bus, mainloop
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    spi = spidev.SpiDev()

    adapter_path = find_adapter(bus)
    if not adapter_path:
        print('BLE adapter not found')
        return

    service_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), GATT_MANAGER_IFACE)
    advertisement_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), LE_ADVERTISING_MANAGER_IFACE)

    app = Application(bus, spi)
    adv = Advertisement(bus, 0, 'peripheral')

    mainloop = GLib.MainLoop()

    # Register GATT application
    service_manager.RegisterApplication(
        app.get_path(),
        {},
        reply_handler=lambda: print('GATT app registered'),
        error_handler=lambda error: print('Failed to register app:', error)
    )

    # Register advertisement
    advertisement_manager.RegisterAdvertisement(
        adv.get_path(),
        {},
        reply_handler=lambda: print('Advertisement registered'),
        error_handler=lambda error: print('Failed to register advertisement:', error)
    )

    try:
        mainloop.run()
    except KeyboardInterrupt:
        print('Stopping...')
        GPIO.cleanup()
        spi.close()
        sys.exit()

if __name__ == '__main__':
    main()
