import os
import json
import cv2
import time
import rerun as rr
import rerun.blueprint as rrb
from datetime import datetime
import numpy as np
from numbers import Number
os.environ["RUST_LOG"] = "error"

class RerunEpisodeReader:
    def __init__(self, task_dir = ".", json_file="data.json"):
        self.task_dir = task_dir
        self.json_file = json_file

    def return_episode_data(self, episode_idx):
        # Load episode data on-demand
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_idx:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Episode {episode_idx} data.json not found.")

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            json_file = json.load(jsonf)

        episode_data = []

        # Loop over the data entries and process each one
        for item_data in json_file['data']:
            # Process images and other data
            colors = self._process_images(item_data, 'colors', episode_dir)
            depths = self._process_images(item_data, 'depths', episode_dir)
            audios = self._process_audio(item_data, 'audios', episode_dir)

            # Append the data in the item_data list
            episode_data.append(
                {
                    'idx': item_data.get('idx', 0),
                    'colors': colors,
                    'depths': depths,
                    'states': item_data.get('states', {}),
                    'actions': item_data.get('actions', {}),
                    'tactiles': item_data.get('tactiles', {}),
                    'audios': audios,
                }
            )

        return episode_data

    def _process_images(self, item_data, data_type, dir_path):
        images = {}

        for key, file_name in item_data.get(data_type, {}).items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        continue
                    if image.ndim == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    elif image.ndim == 3:
                        if image.shape[2] == 3:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        elif image.shape[2] == 4:
                            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                    images[key] = image
        return images

    def _process_audio(self, item_data, data_type, episode_dir):
        audio_data = {}
        dir_path = os.path.join(episode_dir, data_type)

        for key, file_name in item_data.get(data_type, {}).items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    pass  # Handle audio data if needed
        return audio_data

class RerunLogger:
    def __init__(self, prefix="", IdxRangeBoundary=30, memory_limit=None):
        self.prefix = prefix
        self.IdxRangeBoundary = IdxRangeBoundary
        self._blueprint_initialized = False
        self._invalid_image_keys_warned = set()
        rr.init(datetime.now().strftime("Runtime_%Y%m%d_%H%M%S"))
        if memory_limit:
            rr.spawn(memory_limit=memory_limit, hide_welcome_screen=True)
        else:
            rr.spawn(hide_welcome_screen=True)

    def _flatten_numeric(self, base_path, data):
        """Yield (path, scalar) pairs from nested lists/dicts/arrays."""
        if data is None:
            return
        if isinstance(data, np.generic):
            yield base_path, float(data)
            return
        if isinstance(data, Number):
            yield base_path, float(data)
            return
        if isinstance(data, (list, tuple)):
            for idx, value in enumerate(data):
                yield from self._flatten_numeric(f"{base_path}/{idx}", value)
            return
        if isinstance(data, dict):
            for key, value in data.items():
                yield from self._flatten_numeric(f"{base_path}/{key}", value)
            return
        if hasattr(data, "tolist"):
            yield from self._flatten_numeric(base_path, data.tolist())

    def _infer_time_series_origins(self, item_data):
        origins = set()
        states = item_data.get('states', {}) or {}
        for part, value in states.items():
            if part == "body":
                origins.update(self._body_group_origins("states", value))
            else:
                origins.add(f"{self.prefix}states/{part}")
        actions = item_data.get('actions', {}) or {}
        for part, value in actions.items():
            if part == "body":
                origins.update(self._body_group_origins("actions", value))
            else:
                origins.add(f"{self.prefix}actions/{part}")
        return sorted(origins)

    def _body_group_origins(self, block_name, body_data):
        origins = set()
        if not isinstance(body_data, dict):
            return origins
        for key in body_data.keys():
            group = self._categorize_body_key(key)
            origins.add(f"{self.prefix}{block_name}/body/{group}")
        return origins

    @staticmethod
    def _categorize_body_key(key: str) -> str:
        key_lower = key.lower()
        if "waist" in key_lower:
            return "waist"
        if "move" in key_lower:
            return "move"
        if "lift" in key_lower or "height" in key_lower:
            return "lift"
        return "other"

    def _infer_image_origins(self, item_data):
        color_origins = []
        depth_origins = []
        colors = item_data.get('colors', {}) or {}
        for color_key in colors.keys():
            color_origins.append(f"{self.prefix}colors/{color_key}")
        depths = item_data.get('depths', {}) or {}
        for depth_key in depths.keys():
            depth_origins.append(f"{self.prefix}depths/{depth_key}")
        return color_origins, depth_origins

    def _setup_dynamic_blueprint(self, item_data):
        if self._blueprint_initialized or not self.IdxRangeBoundary:
            self._blueprint_initialized = True
            return

        timeseries_origins = self._infer_time_series_origins(item_data)
        color_origins, depth_origins = self._infer_image_origins(item_data)

        views = []
        for origin in timeseries_origins:
            views.append(
                rrb.TimeSeriesView(
                    origin=origin,
                    time_ranges=[
                        rrb.VisibleTimeRange(
                            "idx",
                            start=rrb.TimeRangeBoundary.cursor_relative(seq=-self.IdxRangeBoundary),
                            end=rrb.TimeRangeBoundary.cursor_relative(),
                        )
                    ],
                    plot_legend=rrb.PlotLegend(visible=True),
                )
            )

        for origin in color_origins + depth_origins:
            views.append(
                rrb.Spatial2DView(
                    origin=origin,
                    time_ranges=[
                        rrb.VisibleTimeRange(
                            "idx",
                            start=rrb.TimeRangeBoundary.cursor_relative(seq=-self.IdxRangeBoundary),
                            end=rrb.TimeRangeBoundary.cursor_relative(),
                        )
                    ],
                )
            )

        if views:
            grid = rrb.Grid(
                contents=views,
                grid_columns=min(2, len(views)),
            )
            blueprint = rrb.Blueprint(
                grid,
                rr.blueprint.SelectionPanel(state=rrb.PanelState.Collapsed),
                rr.blueprint.TimePanel(state=rrb.PanelState.Collapsed),
            )
            rr.send_blueprint(blueprint)

        self._blueprint_initialized = True

    def _log_numeric_block(self, block_name, block_data):
        if not block_data:
            return
        remaining_data = block_data
        if isinstance(block_data, dict) and "body" in block_data:
            self._log_body_groups(block_name, block_data.get("body"))
            remaining_data = {k: v for k, v in block_data.items() if k != "body"}
        base_path = f"{self.prefix}{block_name}"
        for path, value in self._flatten_numeric(base_path, remaining_data):
            if value is None or not np.isfinite(value):
                continue
            rr.log(path, rr.Scalar(value))

    def _log_body_groups(self, block_name, body_data):
        if not isinstance(body_data, dict):
            return
        for key, value in body_data.items():
            group = self._categorize_body_key(key)
            base_path = f"{self.prefix}{block_name}/body/{group}/{key}"
            for path, scalar in self._flatten_numeric(base_path, value):
                if scalar is None or not np.isfinite(scalar):
                    continue
                rr.log(path, rr.Scalar(scalar))

    def _log_images(self, block_name, images_dict):
        if not images_dict:
            return
        for image_key, image_value in images_dict.items():
            if image_value is None:
                continue
            origin = f"{self.prefix}{block_name}/{image_key}"
            if isinstance(image_value, (list, tuple)):
                image_value = np.asarray(image_value)
            if not isinstance(image_value, np.ndarray):
                if origin not in self._invalid_image_keys_warned:
                    rr.log(f"{origin}/warning", rr.TextLog(f"Skipping non-array image payload type={type(image_value)}"))
                    self._invalid_image_keys_warned.add(origin)
                continue
            if image_value.size == 0 or image_value.ndim not in (2, 3):
                if origin not in self._invalid_image_keys_warned:
                    rr.log(f"{origin}/warning", rr.TextLog(f"Skipping image with invalid shape {image_value.shape}"))
                    self._invalid_image_keys_warned.add(origin)
                continue
            origin = f"{self.prefix}{block_name}/{image_key}"
            rr.log(origin, rr.Image(image_value))

    def log_item_data(self, item_data: dict):
        if not self._blueprint_initialized:
            self._setup_dynamic_blueprint(item_data)

        rr.set_time_sequence("idx", item_data.get('idx', 0))

        self._log_numeric_block("states", item_data.get('states', {}) or {})
        self._log_numeric_block("actions", item_data.get('actions', {}) or {})
        self._log_images("colors", item_data.get('colors', {}) or {})
        self._log_images("depths", item_data.get('depths', {}) or {})

        # Placeholder hooks for tactile/audio extensions
        # tactiles = item_data.get('tactiles', {}) or {}
        # audios = item_data.get('audios', {}) or {}

    def log_episode_data(self, episode_data: list):
        for item_data in episode_data:
            self.log_item_data(item_data)


if __name__ == "__main__":
    import gdown
    import zipfile
    import os
    import logging_mp
    logger_mp = logging_mp.getLogger(__name__)

    zip_file = "rerun_testdata.zip"
    zip_file_download_url = "https://drive.google.com/file/d/1f5UuFl1z_gaByg_7jDRj1_NxfJZh2evD/view?usp=sharing"
    unzip_file_output_dir = "./testdata"
    if not os.path.exists(os.path.join(unzip_file_output_dir, "episode_0006")):
        if not os.path.exists(zip_file):
            file_id = zip_file_download_url.split('/')[5]
            gdown.download(id=file_id, output=zip_file, quiet=False)
            logger_mp.info("download ok.")
        if not os.path.exists(unzip_file_output_dir):
            os.makedirs(unzip_file_output_dir)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_file_output_dir)
        logger_mp.info("uncompress ok.")
        os.remove(zip_file)
        logger_mp.info("clean file ok.")
    else:
        logger_mp.info("rerun_testdata exits.")


    episode_reader = RerunEpisodeReader(task_dir = unzip_file_output_dir)
    # TEST EXAMPLE 1 : OFFLINE DATA TEST
    user_input = input("Please enter the start signal (enter 'off' or 'on' to start the subsequent program):\n")
    if user_input.lower() == 'off':
        episode_data6 = episode_reader.return_episode_data(6)
        logger_mp.info("Starting offline visualization...")
        offline_logger = RerunLogger(prefix="offline/")
        offline_logger.log_episode_data(episode_data6)
        logger_mp.info("Offline visualization completed.")

    # TEST EXAMPLE 2 : ONLINE DATA TEST, SLIDE WINDOW SIZE IS 60, MEMORY LIMIT IS 50MB
    if user_input.lower() == 'on':
        episode_data8 = episode_reader.return_episode_data(8)
        logger_mp.info("Starting online visualization with fixed idx size...")
        online_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit='50MB')
        for item_data in episode_data8:
            online_logger.log_item_data(item_data)
            time.sleep(0.033) # 30hz
        logger_mp.info("Online visualization completed.")


    # # TEST DATA OF data_dir
    # data_dir = "./data"
    # episode_data_number = 10
    # episode_reader2 = RerunEpisodeReader(task_dir = data_dir)
    # user_input = input("Please enter the start signal (enter 'on' to start the subsequent program):\n")
    # episode_data8 = episode_reader2.return_episode_data(episode_data_number)
    # if user_input.lower() == 'on':
    #     # Example 2: Offline Visualization with Fixed Time Window
    #     logger_mp.info("Starting offline visualization with fixed idx size...")
    #     online_logger = RerunLogger(prefix="offline/", IdxRangeBoundary = 60)
    #     for item_data in episode_data8:
    #         online_logger.log_item_data(item_data)
    #         time.sleep(0.033) # 30hz
    #     logger_mp.info("Offline visualization completed.")
