import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from calc_conc_baseline_fitting_GUI import calc_conc_baseline_fitting_GUI_Sim
import numpy as np
import threading
import asyncio
from bleak import BleakClient
from bleak import BleakScanner
import time
import traceback  # Import traceback for detailed exception information
import sys  # Import sys to redirect stdout

# Bluetooth address of the Raspberry Pi
#raspberry_pi_address = "4BD3D63F-5A19-AF31-C423-9EA0E3C656E6"
raspberry_pi_address = "0F9104FF-A777-7A55-CE54-78D65973ABFB"

# Replace with the appropriate UUIDs for the BLE characteristic and service
CHARACTERISTIC_UUID = "70c4f540-3725-49da-b0bd-3f6ff779d6b5"
SERVICE_UUID = "d647a4b6-a14c-4695-b517-c83c1636617b"

# Create the asyncio event loop and start it in a new thread
loop = asyncio.new_event_loop()

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

t = threading.Thread(target=start_loop, args=(loop,), daemon=True)
t.start()

class IbsenCalcGUI:
    def __init__(self, root, loop):
        self.root = root
        self.loop = loop
        self.root.title("Mini Real-time Enhanced Neurometabolic Assessment System GUI")
        self.ref_spectra = None  # Initialize reference spectra as None
        self.init_gui()
        self.running = False
        self.bt_client = None  # Bluetooth client for communication
        self.device_connected = False  # Track connection status

        self.time_data = []
        self.hbo2_data = []
        self.hhb_data = []
        self.cco_data = []
        self.event_flags = []  # Track whether an event occurred at each time point (0 or 1)
        self.event_details_data = []  # Track event details for each time point (empty or event text)
        self.event_times = []  # List to store event times
        self.event_texts = []  # List to store event details
        self.output_file = None  # Output file for saving results
        self.mean_count = 0
        self.temperature = None  # Initialize temperature display
        self.expecting_reference = False  # Flag to indicate if expecting reference data

        # Initialize variables for initial concentrations
        self.initial_hbo2 = None
        self.initial_hhb = None
        self.initial_cco = None

        # Initialize data buffers for reassembly
        self.received_chunks = {}
        self.expected_total_chunks = None
        self.receiving_data = False
        self.current_packet_type = None

        # Add a message Label at the bottom of the GUI
        self.message_label = tk.Label(self.root, text="", anchor='w')
        self.message_label.grid(row=3, column=0, columnspan=2, sticky='we')

        # Redirect sys.stdout to the PrintLogger
        sys.stdout = PrintLogger(self.message_label)

    def init_gui(self):
        """Initialize the GUI with tabs and controls."""
        # Configure grid to manage space between control panel and tabs
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Controls panel on the left
        controls_frame = tk.Frame(self.root, width=200)
        controls_frame.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")

        self.start_button = tk.Button(controls_frame, text="Start", bg="green", height=1, width=15, command=self.start_analysis)
        self.start_button.grid(row=0, column=0, pady=5)

        self.stop_button = tk.Button(controls_frame, text="Stop", bg="red", height=1, width=15, command=self.stop)
        self.stop_button.grid(row=1, column=0, pady=5)

        self.mark_event_button = tk.Button(controls_frame, text="Mark Event", bg="yellow", height=1, width=15, command=self.mark_event)
        self.mark_event_button.grid(row=2, column=0, pady=5)

        # Integration Time on the side panel
        self.integration_time_label = tk.Label(controls_frame, text="Integration Time (ms)")
        self.integration_time_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")

        self.integration_time = tk.Entry(controls_frame)
        self.integration_time.insert(0, "1000")  # Default value
        self.integration_time.grid(row=3, column=1, padx=5, pady=5)
        self.integration_time.bind("<Return>", self.update_integration_time)  # Triggers on pressing Enter

        # Event Details text box (below Integration Time)
        self.event_details_label = tk.Label(controls_frame, text="Event Details")
        self.event_details_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")

        self.event_details_entry = tk.Entry(controls_frame)
        self.event_details_entry.grid(row=4, column=1, padx=5, pady=5)

        # Measure Reference Button
        self.measure_ref_button = tk.Button(controls_frame, text="Measure Reference", command=self.measure_reference)
        self.measure_ref_button.grid(row=5, column=0, padx=5, pady=5)

        # Bluetooth Connect Button
        self.bt_connect_button = tk.Button(controls_frame, text="Connect Bluetooth", command=self.connect_bluetooth)
        self.bt_connect_button.grid(row=6, column=0, pady=5)

        # Bluetooth connection status (circle indicator)
        self.bt_status_canvas = tk.Canvas(controls_frame, width=25, height=25)
        self.bt_status_circle = self.bt_status_canvas.create_oval(5, 5, 25, 25, fill="red")
        self.bt_status_canvas.grid(row=6, column=1, padx=5)

        # Status message area with aligned circle and text
        status_frame = tk.Frame(self.root)
        status_frame.grid(row=1, column=0, padx=5, pady=5, sticky="sw")

        # Status text ("Running" or "Stopped")
        self.status_label = tk.Label(status_frame, text="Stopped", fg="red")
        self.status_label.grid(row=0, column=0, padx=(0, 5), sticky="e")

        # Canvas for the status circle (align with the text)
        self.status_canvas = tk.Canvas(status_frame, width=25, height=25)
        self.status_circle = self.status_canvas.create_oval(5, 5, 25, 25, fill="red")
        self.status_canvas.grid(row=0, column=1, sticky="w")

        # Temperature display
        self.temperature_label = tk.Label(self.root, text="Temperature: -- °C", fg="blue")
        self.temperature_label.grid(row=2, column=1, padx=5, pady=5, sticky="se")

        # Main tab view for Settings, Spectra, Concentrations, etc.
        self.tab_control = ttk.Notebook(self.root)
        self.tab_control.grid(row=0, column=1, padx=5, pady=10, sticky="nsew")

        # Settings Tab
        self.settings_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.settings_tab, text="Settings")

        # Pathlength section in Settings
        pathlength_frame = tk.LabelFrame(self.settings_tab, text="Pathlength")
        pathlength_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=10, sticky="nsew")

        # Age Input
        tk.Label(pathlength_frame, text="Enter Age:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.age_entry = tk.Entry(pathlength_frame)
        self.age_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # DPF Selection
        tk.Label(pathlength_frame, text="Select DPF Value:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.dpf_var = tk.StringVar(value="Select Measurement Area")
        self.dpf_dropdown = ttk.Combobox(pathlength_frame, textvariable=self.dpf_var)
        self.dpf_dropdown['values'] = ("1. Adult Forearm 4.16", "2. Baby Head 4.99", "3. Adult Head 6.26", "4. Adult Leg 5.51")
        self.dpf_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(pathlength_frame, text="Separation (cm)").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.separation = tk.Entry(pathlength_frame)
        self.separation.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(pathlength_frame, text="Water Fraction").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.water_fraction = tk.Entry(pathlength_frame)
        self.water_fraction.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        # Save Output Location
        tk.Label(self.settings_tab, text="Save Output:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.output_path_var = tk.StringVar()
        self.output_path_entry = tk.Entry(self.settings_tab, textvariable=self.output_path_var)
        self.output_path_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.browse_output_button = tk.Button(self.settings_tab, text="Browse", command=self.browse_output)
        self.browse_output_button.grid(row=2, column=2, padx=5, pady=5)

        # Spectra Tab (Live spectra graph)
        self.spectra_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.spectra_tab, text="Spectra")

        # Spectra plot sized to fit the tab, accounting for the side panel
        self.fig_spectra, self.ax_spectra = plt.subplots(figsize=(5, 3))
        self.spectra_plot, = self.ax_spectra.plot([], [], label="Live Spectra", linewidth=1)
        self.ax_spectra.grid(True)

        self.ax_spectra.set_xlabel('Wavelength (nm)', fontsize=8)
        self.ax_spectra.set_ylabel('Intensity', fontsize=8)
        self.ax_spectra.tick_params(axis='both', labelsize=8)
        self.ax_spectra.legend(fontsize=8)  # Adjust the legend font size for the intensity spectra plot
        self.ax_spectra.set_xlim([400, 1000])
        self.ax_spectra.set_ylim([0, 66000])

        self.canvas_spectra = FigureCanvasTkAgg(self.fig_spectra, master=self.spectra_tab)
        self.canvas_spectra.draw()
        self.canvas_spectra.get_tk_widget().grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        # Set row and column weights to allow resizing
        self.spectra_tab.grid_rowconfigure(0, weight=1)
        self.spectra_tab.grid_columnconfigure(0, weight=1)

        # Concentrations Tab (Resized to fit properly)
        self.concentrations_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.concentrations_tab, text='Concentrations')

        # Adjust figure size to ensure that the graph fits in the tab area
        self.fig_concentrations, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 4))
        self.fig_concentrations.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)

        # Initialize empty plots with reduced line width
        self.hbo2_plot, = self.ax1.plot([], [], 'r', label='HbO2', linewidth=1)
        self.hhb_plot, = self.ax1.plot([], [], 'b', label='HHb', linewidth=1)
        self.cco_plot, = self.ax2.plot([], [], 'g', label='CCO', linewidth=1)

        self.ax1.set_ylabel('ΔHbO2 and ΔHHb (µM)', fontsize=8)
        self.ax1.set_xlabel('Time (s)', fontsize=8)
        self.ax1.tick_params(axis='both', labelsize=8)

        self.ax2.set_ylabel('ΔCCO (µM)', fontsize=8)
        self.ax2.set_xlabel('Time (s)', fontsize=8)
        self.ax2.tick_params(axis='both', labelsize=8)

        self.ax1.legend(fontsize=8)
        self.ax1.grid(True)

        self.ax2.legend(fontsize=8)
        self.ax2.grid(True)

        # Embed plot into the Concentrations tab
        self.canvas_concentrations = FigureCanvasTkAgg(self.fig_concentrations, master=self.concentrations_tab)
        self.canvas_concentrations.draw()
        self.canvas_concentrations.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Set row and column weights to allow resizing
        self.concentrations_tab.grid_rowconfigure(0, weight=1)
        self.concentrations_tab.grid_columnconfigure(0, weight=1)

        # Reference Tab
        self.reference_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.reference_tab, text="Reference")

        # Reference spectra plot
        self.fig_reference, self.ax_reference = plt.subplots(figsize=(5, 3))
        self.reference_plot, = self.ax_reference.plot([], [], label="Reference Spectra", linewidth=1)
        self.ax_reference.grid(True)

        self.ax_reference.set_xlabel('Wavelength (nm)', fontsize=8)
        self.ax_reference.set_ylabel('Intensity', fontsize=8)
        self.ax_reference.tick_params(axis='both', labelsize=8)
        self.ax_reference.legend(fontsize=8)

        self.ax_reference.set_xlim([400, 1000])
        self.ax_reference.set_ylim([0, 66000])

        self.canvas_reference = FigureCanvasTkAgg(self.fig_reference, master=self.reference_tab)
        self.canvas_reference.draw()
        self.canvas_reference.get_tk_widget().grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        # Set row and column weights to allow resizing
        self.reference_tab.grid_rowconfigure(0, weight=1)
        self.reference_tab.grid_columnconfigure(0, weight=1)

        # Placeholder tabs
        for tab_name in ["Tissue Saturation"]:
            new_tab = ttk.Frame(self.tab_control)
            self.tab_control.add(new_tab, text=tab_name)

        self.tab_control.grid(row=0, column=1, padx=5, pady=10, sticky="nsew")

    def connect_bluetooth(self):
        """Connect to the Raspberry Pi using Bluetooth."""
        future = asyncio.run_coroutine_threadsafe(self._connect_bluetooth(), self.loop)
        future.add_done_callback(self._on_connect_complete)

    async def _connect_bluetooth(self):
        """Asynchronous method to handle Bluetooth connection."""
        try:
            print(f"Attempting to connect to Raspberry Pi at {raspberry_pi_address}...")
            self.bt_client = BleakClient(raspberry_pi_address, loop=self.loop)
            await self.bt_client.connect()

            if self.bt_client.is_connected:
                print(f"Connected to Raspberry Pi at {raspberry_pi_address}")
                self.device_connected = True
                # Schedule GUI updates in the main thread
                self.root.after(0, lambda: self.bt_status_canvas.itemconfig(self.bt_status_circle, fill="green"))

                # Ensure services are populated
                await self.bt_client.get_services()
                services = self.bt_client.services
                print("Available services and characteristics:")
                found_characteristic = False
                for service in services:
                    print(f"Service UUID: {service.uuid}")
                    if service.uuid == SERVICE_UUID:
                        for characteristic in service.characteristics:
                            print(f"Characteristic UUID: {characteristic.uuid}, Properties: {characteristic.properties}")
                            if characteristic.uuid == CHARACTERISTIC_UUID:
                                print(f"Custom characteristic {CHARACTERISTIC_UUID} found!")
                                found_characteristic = True
                                break
                if not found_characteristic:
                    print(f"Custom characteristic {CHARACTERISTIC_UUID} not found.")
                    self.root.after(0, lambda: self.bt_status_canvas.itemconfig(self.bt_status_circle, fill="red"))
                else:
                    # Start notifications
                    await self.bt_client.start_notify(CHARACTERISTIC_UUID, self.notification_handler)
            else:
                print(f"Failed to connect to Raspberry Pi at {raspberry_pi_address}")
                self.root.after(0, lambda: self.bt_status_canvas.itemconfig(self.bt_status_circle, fill="red"))
        except Exception as e:
            print(f"Could not connect to device: {e}")
            self.root.after(0, lambda: self.bt_status_canvas.itemconfig(self.bt_status_circle, fill="red"))

    def _on_connect_complete(self, future):
        try:
            future.result()
        except Exception as e:
            print(f"Error during connection: {e}")

    def measure_reference(self):
        """Send 'Reference' message to Raspberry Pi and handle incoming reference spectra."""
        if not self.device_connected:
            messagebox.showerror("Error", "No Bluetooth device connected.")
            return
        self.expecting_reference = True  # Set flag to expect reference data
        future = asyncio.run_coroutine_threadsafe(self.send_bluetooth_message("Reference"), self.loop)
        future.add_done_callback(lambda f: print("Reference message sent"))

    def notification_handler(self, sender, data):
        """Handle incoming notifications and reassemble data."""
        try:
            packet_type = data[0]

            if packet_type in [0x01, 0x03]:  # Data Chunk or Reference Data Chunk
                chunk_number = data[1]
                total_chunks = data[2]
                data_chunk = data[3:]

                if not self.receiving_data or self.current_packet_type != packet_type:
                    # Starting a new data transmission
                    self.receiving_data = True
                    self.expected_total_chunks = total_chunks
                    self.received_chunks = {}
                    self.current_packet_type = packet_type

                self.received_chunks[chunk_number] = data_chunk

                if len(self.received_chunks) == self.expected_total_chunks:
                    # Reassemble the data
                    full_data = bytearray()
                    for i in range(self.expected_total_chunks):
                        full_data += self.received_chunks[i]

                    # Reset for next data transmission
                    self.received_chunks = {}
                    self.expected_total_chunks = None
                    self.receiving_data = False
                    self.current_packet_type = None

                    # Process the full data
                    if packet_type == 0x01:
                        # Schedule processing in the main thread
                        self.root.after(0, self.process_full_data, full_data)
                    elif packet_type == 0x03:
                        # Schedule processing in the main thread
                        self.root.after(0, self.process_reference_data, full_data)

            elif packet_type == 0x02:  # Control Message
                message = data[1:].decode('utf-8')
                print(f"Received control message: {message}")
                # Handle control messages if needed

        except Exception as e:
            print(f"Error in notification handler: {e}")

    def process_full_data(self, data):
        """Process the reassembled spectral and temperature data."""
        try:
            NUM_PIXELS = 256  # Update according to your detector
            BYTES_PER_PIXEL = 2
            total_spectral_bytes = NUM_PIXELS * BYTES_PER_PIXEL
            spectral_data = data[:total_spectral_bytes]
            temperature_data = data[total_spectral_bytes:]

            intensity_values = np.frombuffer(spectral_data, dtype=">u2")
            intensity_values = np.clip(intensity_values, 0, 65535)  # Ensure values stay within 16-bit range

            if len(temperature_data) >= 2:
                temp_value_raw = int.from_bytes(temperature_data[:2], byteorder='big', signed=False)
                temp_celsius = self.convert_temperature(temp_value_raw)
                self.temperature = temp_celsius
                self.temperature_label.config(text=f"Temperature: {self.temperature:.2f} °C")
            else:
                self.temperature_label.config(text="Temperature: N/A")

            pix = np.arange(len(intensity_values))

            # Compute wavelengths using the calibration coefficients
            B0, B1, B2, B3, B4 = 1.10314E+03, -1.93566E+00, -2.46874E-03, 1.85252E-06, -5.21135E-09
            wavelengths = B0 + B1 * pix + B2 * pix**2 + B3 * pix**3 + B4 * pix**4

            if self.ref_spectra is None:
                print("No reference spectra available. Please measure reference first.")
                return

            # Trim arrays to length 255 to match extinction coefficients
            intensity_values = intensity_values[:255]
            wavelengths = wavelengths[:255]
            self.ref_spectra = self.ref_spectra[:255]

            # Convert to int32 before subtraction to prevent overflow
            corrected_spectra = intensity_values.astype(np.int32) - self.ref_spectra.astype(np.int32)
            corrected_spectra = np.clip(corrected_spectra, 0, 65535)  # Clamp values to prevent overflow

            # Optional: Apply smoothing
            smoothed_spectra = np.convolve(corrected_spectra, np.ones(5) / 5, mode='same')

            # Debugging information
            print("Intensity values - Min:", np.min(intensity_values), "Max:", np.max(intensity_values))
            print("Reference spectra - Min:", np.min(self.ref_spectra), "Max:", np.max(self.ref_spectra))
            print("Corrected spectra - Min:", np.min(corrected_spectra), "Max:", np.max(corrected_spectra))

            # Update live spectra plot
            self.spectra_plot.set_data(wavelengths, smoothed_spectra)
            self.ax_spectra.relim()
            self.ax_spectra.autoscale_view()
            self.canvas_spectra.draw()
            print("Live spectra received and plotted")

            # Process live data to calculate concentrations and update plots
            self.process_live_data(wavelengths, smoothed_spectra)

        except Exception as e:
            print(f"Error in process_full_data: {e}")
            traceback.print_exc()

    def process_reference_data(self, data):
        """Process the reassembled reference spectral data."""
        try:
            NUM_PIXELS = 256  # Update according to your detector
            BYTES_PER_PIXEL = 2
            total_spectral_bytes = NUM_PIXELS * BYTES_PER_PIXEL
            spectral_data = data[:total_spectral_bytes]
            temperature_data = data[total_spectral_bytes:]

            intensity_values = np.frombuffer(spectral_data, dtype=">u2")
            intensity_values = np.clip(intensity_values, 0, 65535)  # Ensure values stay within 16-bit range

            if len(temperature_data) >= 2:
                temp_value_raw = int.from_bytes(temperature_data[:2], byteorder='big', signed=False)
                temp_celsius = self.convert_temperature(temp_value_raw)
                self.temperature = temp_celsius
                self.temperature_label.config(text=f"Temperature: {self.temperature:.2f} °C")
            else:
                self.temperature_label.config(text="Temperature: N/A")

            pix = np.arange(len(intensity_values))

            # Compute wavelengths using the calibration coefficients
            B0, B1, B2, B3, B4 = 1.10314E+03, -1.93566E+00, -2.46874E-03, 1.85252E-06, -5.21135E-09
            wavelengths = B0 + B1 * pix + B2 * pix**2 + B3 * pix**3 + B4 * pix**4

            # Trim arrays to length 255 to match extinction coefficients
            intensity_values = intensity_values[:255]
            wavelengths = wavelengths[:255]

            # Set the received data as reference spectra
            self.ref_spectra = intensity_values.copy()

            # Update reference spectra plot
            self.reference_plot.set_data(wavelengths, self.ref_spectra)
            self.ax_reference.relim()
            self.ax_reference.autoscale_view()
            self.canvas_reference.draw()
            print("Reference spectra received and plotted")

            self.expecting_reference = False  # Reset flag

        except Exception as e:
            print(f"Error in process_reference_data: {e}")
            traceback.print_exc()

    def convert_temperature(self, raw_value):
        """Convert raw temperature value to Celsius using the spectrometer's data sheet."""

        # ADC counts to temperature mapping from the data sheet
        adc_to_temp_table = [
            (3977, -40),
            (3933, -35),
            (3876, -30),
            (3803, -25),
            (3712, -20),
            (3601, -15),
            (3468, -10),
            (3312, -5),
            (3135, 0),
            (2938, 5),
            (2726, 10),
            (2503, 15),
            (2275, 20),
            (2048, 25),
            (1828, 30),
            (1618, 35),
            (1422, 40),
            (1246, 45),
            (1084, 50),
            (940, 55),
            (814, 60),
            (705, 65),
            (610, 70),
            (528, 75),
            (455, 80),
            (396, 85),
            (342, 90),
            (296, 95),
            (257, 100),
            (225, 105),
            (199, 110),
            (173, 115),
            (150, 120),
            (135, 125),
        ]

        # Extract ADC counts and temperatures into separate lists
        adc_counts = [entry[0] for entry in adc_to_temp_table]
        temperatures = [entry[1] for entry in adc_to_temp_table]

        # If the raw_value is outside the range, return the closest temperature
        if raw_value >= adc_counts[0]:
            return temperatures[0]
        elif raw_value <= adc_counts[-1]:
            return temperatures[-1]
        else:
            # Interpolate between the closest values
            for i in range(len(adc_counts) - 1):
                if adc_counts[i] >= raw_value >= adc_counts[i + 1]:
                    # Linear interpolation
                    x0, y0 = adc_counts[i], temperatures[i]
                    x1, y1 = adc_counts[i + 1], temperatures[i + 1]
                    temp = y0 + (raw_value - x0) * (y1 - y0) / (x1 - x0)
                    return temp

        # If no match found, return None
        return None

    def start_analysis(self):
        """Start the analysis by receiving live spectra data."""
        if not self.device_connected:
            messagebox.showerror("Error", "No Bluetooth device connected.")
            return
        if self.ref_spectra is None:
            messagebox.showerror("Error", "Reference spectra not measured. Please measure reference first.")
            return

        # Update the status to "Running"
        self.status_label.config(text="Running", fg="green")
        self.status_canvas.itemconfig(self.status_circle, fill="green")
        self.running = True
        self.start_time = time.time()  # Record the start time
        print("Analysis started")

        # Send the 'start' message to the Raspberry Pi
        future = asyncio.run_coroutine_threadsafe(self.send_bluetooth_message("start"), self.loop)
        future.add_done_callback(self._on_analysis_started)

    def _on_analysis_started(self, future):
        try:
            future.result()
            print("Analysis started")
        except Exception as e:
            print(f"Error starting analysis: {e}")

    def stop(self):
        """Stop data acquisition."""
        # Update the status to "Stopped"
        self.status_label.config(text="Stopped", fg="red")
        self.status_canvas.itemconfig(self.status_circle, fill="red")
        self.running = False
        print("Analysis stopped")

        # Send the 'stop' message to the Raspberry Pi
        future = asyncio.run_coroutine_threadsafe(self.send_bluetooth_message("stop"), self.loop)
        future.add_done_callback(self._on_analysis_stopped)

    def _on_analysis_stopped(self, future):
        try:
            future.result()
            print("Analysis stopped")
        except Exception as e:
            print(f"Error stopping analysis: {e}")

    def update_integration_time(self, event=None):
        """Update the integration time and send it via Bluetooth."""
        integration_time = self.integration_time.get().strip()
        if integration_time and self.device_connected:
            try:
                # Ensure integration_time is in milliseconds
                integration_time_ms = float(integration_time)
                future = asyncio.run_coroutine_threadsafe(
                    self.send_bluetooth_message(f"integration_time:{integration_time_ms}"), self.loop
                )
                future.add_done_callback(lambda f: print(f"Integration time set to {integration_time_ms} ms"))
            except Exception as e:
                print(f"Failed to send integration time: {e}")

    async def send_bluetooth_message(self, message):
        """Send a message via Bluetooth."""
        if self.device_connected and self.bt_client:
            try:
                print(f"Attempting to send message: {message}")
                await self.bt_client.write_gatt_char(CHARACTERISTIC_UUID, message.encode('utf-8'))
                print(f"Message sent: {message}")
            except Exception as e:
                print(f"Error sending message: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error sending message: {e}"))

    def mark_event(self):
        """Mark the event by adding a vertical line and logging details."""
        if not self.running:
            messagebox.showwarning("Warning", "No data is currently being collected.")
            return

        # Get the current time
        current_time = time.time() - self.start_time  # Adjust based on when data collection started

        event_text = self.event_details_entry.get().strip()

        if event_text:
            # Check if this event has already been added
            if current_time not in self.event_times:
                self.event_times.append(current_time)
                self.event_texts.append(event_text)

                # Add a vertical line on both HbO2/HHb and CCO graphs at the correct time
                self.ax1.axvline(x=current_time, color='k', linestyle='--', linewidth=1)
                self.ax2.axvline(x=current_time, color='k', linestyle='--', linewidth=1)

                # Add event text annotation on both graphs at the top
                self.update_event_text(current_time, event_text)

                # Mark the current time point with an event in the output data
                self.event_flags[-1] = 1  # Update the event flag for the most recent time point
                self.event_details_data[-1] = event_text  # Update the event details for the most recent time point

                self.canvas_concentrations.draw()

                # Log the event details in the output file if available
                if self.output_path_var.get():
                    with open(self.output_path_var.get(), 'a') as f:
                        f.write(f"Event at {current_time} seconds: {event_text}\n")
            else:
                messagebox.showwarning("Warning", "This event has already been marked.")
        else:
            messagebox.showwarning("Warning", "Please enter event details before marking.")

    def update_event_text(self, current_time, event_text):
        """Dynamically update event text position to always stay at the top of both graphs."""
        ylim_top_ax1 = self.ax1.get_ylim()[1]  # Get the current upper y-limit for HbO2/HHb graph
        ylim_top_ax2 = self.ax2.get_ylim()[1]  # Get the current upper y-limit for CCO graph
        # Add event text to both graphs
        self.ax1.text(current_time, ylim_top_ax1, event_text, rotation=90, verticalalignment='top', fontsize=6)
        self.ax2.text(current_time, ylim_top_ax2, event_text, rotation=90, verticalalignment='top', fontsize=6)

    def browse_output(self):
        output_file = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        self.output_path_var.set(output_file)

    def process_live_data(self, wavelengths, corrected_spectra):
        """Process the live spectra data and update the plots dynamically."""
        try:
            # Define parameters for concentration calculation
            integration_time = float(self.integration_time.get()) if self.integration_time.get() else 1000
            age = int(self.age_entry.get()) if self.age_entry.get() else 30  # Default age
            if self.dpf_var.get() and self.dpf_var.get() != "Select Measurement Area":
                dpf_value = float(self.dpf_var.get().split()[-1])  # Extract the DPF value
            else:
                dpf_value = 6.26  # Default DPF
            separation = float(self.separation.get()) if self.separation.get() else 3.0  # Default separation

            # Use the full (trimmed) spectra for concentration calculations
            HbO2, HHb, CCO, Time, HbT = calc_conc_baseline_fitting_GUI_Sim(
                self.ref_spectra.flatten(), corrected_spectra.flatten(), wavelengths, integration_time, None, None, age, dpf_value, separation)

            # Store initial concentrations if not already set
            if self.initial_hbo2 is None:
                self.initial_hbo2 = HbO2
                self.initial_hhb = HHb
                self.initial_cco = CCO
                self.start_time = time.time()  # Start time when first data is processed

            # Calculate change in concentration
            delta_HbO2 = HbO2 - self.initial_hbo2
            delta_HHb = HHb - self.initial_hhb
            delta_CCO = CCO - self.initial_cco

            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time

            # Append new data
            self.time_data.append(elapsed_time)
            self.hbo2_data.append(delta_HbO2)
            self.hhb_data.append(delta_HHb)
            self.cco_data.append(delta_CCO)
            self.event_flags.append(0)  # Default value is 0 for "no event"
            self.event_details_data.append('')  # Default empty string for "no event details"

            # Save data to file if output path is set
            if self.output_path_var.get():
                with open(self.output_path_var.get(), 'w') as f:
                    f.write("Time,HbO2,HHb,CCO,Event,Event Details\n")
                    for i in range(len(self.time_data)):
                        f.write(f"{self.time_data[i]},{self.hbo2_data[i]},{self.hhb_data[i]},{self.cco_data[i]},"
                                f"{self.event_flags[i]},{self.event_details_data[i]}\n")

            # Update plots on the Concentrations tab
            self.hbo2_plot.set_data(self.time_data, self.hbo2_data)
            self.hhb_plot.set_data(self.time_data, self.hhb_data)
            self.cco_plot.set_data(self.time_data, self.cco_data)

            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()

            # Redraw the canvas
            self.canvas_concentrations.draw()

        except Exception as e:
            print(f"Error in process_live_data: {e}")
            traceback.print_exc()

class PrintLogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, msg):
        if msg.strip() != '':
            # Schedule the update in the main thread
            self.text_widget.after(0, self.text_widget.config, {'text': msg.strip()})

    def flush(self):
        pass  # No need to flush anything

async def scan_for_devices():
    devices = await BleakScanner.discover()
    for device in devices:
        print(f"Device: {device.name}, Address: {device.address}")

if __name__ == "__main__":
    root = tk.Tk()
    app = IbsenCalcGUI(root, loop)
    root.mainloop()
