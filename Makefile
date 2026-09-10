# =============================================================================
# Cell Displacement Prediction — Data Preparation Pipeline
# =============================================================================
# Usage:
#   make prepare_data              — run all data-preparation steps for SPLIT (default: train)
#   make prepare_data SPLIT=val    — ...for a specific split instead
#   make prepare_data_all          — run all data-preparation steps for every split in ALL_SPLITS
#   make clean                     — remove generated (derived) data for SPLIT
#   make clean_all                 — remove generated (derived) data for every split in ALL_SPLITS
#
# Data layout expected under data/ (raw/ is scaffolded already; you provide the
# actual files):
#   data/raw/raw_csv_files/<split>/*.csv          — one CSV per tracked video
#   data/raw/raw_frames/<split>/<video>/*.jpg     — one subfolder per video, each
#                                                    locally numbered 0.jpg, 1.jpg, ...
#                                                    Subfolder names must share the
#                                                    same leading "<n>_" prefix as the
#                                                    matching CSV (e.g. "1_...") so
#                                                    videos line up across steps.
#
# Generated under data/intermediate/<split>/ (steps 1-4) — everything for one split
# lives under its own folder, so it's easy to trace back:
#   data/intermediate/<split>/<split>.csv                  — merged CSV (step 1)
#   data/intermediate/<split>/masks/                       — position + displacement
#                                                             masks (step 2)
#   data/intermediate/<split>/first_second/{first,second}/ — consecutive frame pairs,
#                                                             globally numbered (step 3)
#   data/intermediate/<split>/padded/                      — padded frames + masks (step 4)
#
# Generated under data/model_data/ (step 5) — the final patches fed into the model:
#   data/model_data/<split>/input/{first,second}/                            — image patches
#   data/model_data/<split>/output/{pos_masks,displ_masks/{displ_x,displ_y}}/ — mask patches
#
# Configure the variables below (or override on the command line, e.g.
# `make prepare_data SPLIT=val`) before running.
# =============================================================================

PYTHON ?= python3

# ─── Split selection ──────────────────────────────────────────────────────────
# NOTE: keep these assignments free of trailing inline comments — GNU Make bakes
# any whitespace before a same-line "#" into the variable's value.
SPLIT      ?= train
ALL_SPLITS ?= train val test

# ─── Paths (derived from SPLIT) ────────────────────────────────────────────────
RAW_CSV_DIR    := data/raw/raw_csv_files/$(SPLIT)
RAW_FRAMES_DIR := data/raw/raw_frames/$(SPLIT)

SPLIT_DIR         := data/intermediate/$(SPLIT)
SPLIT_CSV         := $(SPLIT_DIR)/$(SPLIT).csv
FIRST_SECOND_DIR  := $(SPLIT_DIR)/first_second
MASKS_DIR         := $(SPLIT_DIR)/masks
PADDED_DIR        := $(SPLIT_DIR)/padded
# split_images_to_patches.py appends input/ and output/ subfolders itself.
PATCHES_DIR       := data/model_data/$(SPLIT)

# ─── Image / patch parameters ─────────────────────────────────────────────────
IMG_HEIGHT   := 1104
IMG_WIDTH    := 1376
IMG_CHANNELS := 3
PATCH_SIZE   := 128
IMG_FORMAT   := jpg

# =============================================================================

.PHONY: prepare_data prepare_data_all merge_csv generate_masks first_second \
        pad_images split_patches clean clean_all help

help:
	@echo "Targets: prepare_data, prepare_data_all, merge_csv, generate_masks, first_second,"
	@echo "         pad_images, split_patches, clean, clean_all"
	@echo "Key variables: SPLIT=$(SPLIT) ALL_SPLITS=$(ALL_SPLITS)"

# ─── Run every step for every split in ALL_SPLITS, one split at a time ────────
prepare_data_all:
	@for s in $(ALL_SPLITS); do \
		$(MAKE) prepare_data SPLIT=$$s ; \
	done

# ─── Full data preparation (runs steps 1-5 in order, for SPLIT) ───────────────
prepare_data: merge_csv generate_masks first_second pad_images split_patches

# ─── Step 1: Combine per-video tracking CSVs into one CSV for SPLIT ───────────
# Skips (without erroring) if raw_csv_files/<SPLIT> has no CSVs yet — lets
# `make prepare_data SPLIT=val` etc. no-op cleanly on splits you haven't filled in.
merge_csv:
	@echo "\n>>> Step 1/5: Merging tracking CSVs for split '$(SPLIT)'..."
	@mkdir -p $(SPLIT_DIR)
	@if ls $(RAW_CSV_DIR)/*.csv >/dev/null 2>&1; then \
		$(PYTHON) prepare_data/merge_csv.py $(RAW_CSV_DIR) -o $(SPLIT_CSV) ; \
	else \
		echo "No CSV files found in $(RAW_CSV_DIR) — skipping." ; \
	fi

# ─── Step 2: CSV → position and displacement masks ────────────────────────────
# Skips if Step 1 produced no merged CSV for SPLIT.
generate_masks:
	@echo "\n>>> Step 2/5: Generating position and displacement masks..."
	@if [ -f $(SPLIT_CSV) ]; then \
		$(PYTHON) prepare_data/generate_tracking_output.py \
			--path_to_csv $(SPLIT_CSV) \
			--output_dir  $(MASKS_DIR) \
			--img_height  $(IMG_HEIGHT) \
			--img_width   $(IMG_WIDTH) ; \
	else \
		echo "No merged CSV at $(SPLIT_CSV) — skipping." ; \
	fi

# ─── Step 3: Raw per-video frame folders → first/second pair folders ──────────
# Skips if raw_frames/<SPLIT> has no per-video subfolders yet.
first_second:
	@echo "\n>>> Step 3/5: Building first/second frame pairs for '$(SPLIT)'..."
	@if find $(RAW_FRAMES_DIR) -mindepth 1 -maxdepth 1 -type d 2>/dev/null | grep -q .; then \
		$(PYTHON) prepare_data/generate_first_second_folder.py \
			$(RAW_FRAMES_DIR) $(FIRST_SECOND_DIR) true ; \
	else \
		echo "No per-video subfolders found in $(RAW_FRAMES_DIR) — skipping." ; \
	fi

# ─── Step 4: Pad images and masks ──────────────────────────────────────────────
# Padding is calculated automatically from img_height, img_width, patch_size.
# Run once per folder: first/, second/, and all three mask types.
# Skips if Steps 2-3 haven't produced masks / first-second frames for SPLIT.
pad_images:
	@echo "\n>>> Step 4/5: Padding images and masks..."
	@if [ -d $(FIRST_SECOND_DIR)/first ] && [ -d $(MASKS_DIR)/pos_masks ]; then \
		$(PYTHON) prepare_data/pad_images.py \
			--input_dir    $(FIRST_SECOND_DIR)/first \
			--output_dir   $(PADDED_DIR)/padded_first \
			--img_height   $(IMG_HEIGHT) \
			--img_width    $(IMG_WIDTH) \
			--patch_size   $(PATCH_SIZE) \
			--img_channels $(IMG_CHANNELS) \
			--format       $(IMG_FORMAT) ; \
		$(PYTHON) prepare_data/pad_images.py \
			--input_dir    $(FIRST_SECOND_DIR)/second \
			--output_dir   $(PADDED_DIR)/padded_second \
			--img_height   $(IMG_HEIGHT) \
			--img_width    $(IMG_WIDTH) \
			--patch_size   $(PATCH_SIZE) \
			--img_channels $(IMG_CHANNELS) \
			--format       $(IMG_FORMAT) ; \
		$(PYTHON) prepare_data/pad_images.py \
			--input_dir    $(MASKS_DIR)/pos_masks \
			--output_dir   $(PADDED_DIR)/padded_pos_masks \
			--img_height   $(IMG_HEIGHT) \
			--img_width    $(IMG_WIDTH) \
			--patch_size   $(PATCH_SIZE) \
			--img_channels 1 \
			--format       png ; \
		$(PYTHON) prepare_data/pad_images.py \
			--input_dir    $(MASKS_DIR)/displ_masks/displ_x \
			--output_dir   $(PADDED_DIR)/padded_displ_masks/padded_displ_x \
			--img_height   $(IMG_HEIGHT) \
			--img_width    $(IMG_WIDTH) \
			--patch_size   $(PATCH_SIZE) \
			--img_channels $(IMG_CHANNELS) \
			--format       png ; \
		$(PYTHON) prepare_data/pad_images.py \
			--input_dir    $(MASKS_DIR)/displ_masks/displ_y \
			--output_dir   $(PADDED_DIR)/padded_displ_masks/padded_displ_y \
			--img_height   $(IMG_HEIGHT) \
			--img_width    $(IMG_WIDTH) \
			--patch_size   $(PATCH_SIZE) \
			--img_channels $(IMG_CHANNELS) \
			--format       png ; \
	else \
		echo "No first/second frames or masks found for '$(SPLIT)' — skipping padding." ; \
	fi

# ─── Step 5: Split padded images into patches (final model input/output data) ──
# Skips if Step 4 produced no padded frames for SPLIT.
split_patches:
	@echo "\n>>> Step 5/5: Splitting into patches..."
	@if [ -d $(PADDED_DIR)/padded_first ]; then \
		$(PYTHON) prepare_data/split_images_to_patches.py \
			--input_dir    $(PADDED_DIR) \
			--output_dir   $(PATCHES_DIR) \
			--img_height   $(IMG_HEIGHT) \
			--img_width    $(IMG_WIDTH) \
			--patch_size   $(PATCH_SIZE) \
			--img_channels $(IMG_CHANNELS) ; \
	else \
		echo "No padded frames found for '$(SPLIT)' — skipping patch splitting." ; \
	fi

# ─── Clean generated data (keeps raw CSVs and raw frames) ─────────────────────
clean:
	@echo "Removing generated data for split '$(SPLIT)'..."
	rm -rf $(SPLIT_DIR) $(PATCHES_DIR)

clean_all:
	@for s in $(ALL_SPLITS); do \
		$(MAKE) clean SPLIT=$$s ; \
	done
