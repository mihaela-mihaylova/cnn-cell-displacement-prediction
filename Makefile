# =============================================================================
# Cell Displacement Prediction Pipeline
# =============================================================================
# Usage:
#   make prepare_data   — run all data-preparation steps for SPLIT, in order
#   make train          — train the model on SPLIT's patches
#   make predict        — run first/second split + inference on PREDICT_SPLIT
#   make all            — prepare data + train
#   make clean          — remove generated (derived) data for SPLIT/PREDICT_SPLIT
#
# Data layout expected under data/ (already scaffolded in this repo):
#   data/csv/raw_csv_files/<split>/*.csv        — one CSV per tracked video (you provide)
#   data/csv/csv_per_split/<split>.csv          — merged CSV (generated)
#   data/frames/raw_frames/<split>/*.jpg        — raw video frames, numbered 0.jpg, 1.jpg, ...
#                                                  matching the video order in the merged CSV
#                                                  (you provide)
#   data/frames/first_second/<split>/{first,second}/   — consecutive frame pairs (generated)
#   data/frames/model_data/output/<split>/             — position + displacement masks (generated)
#   data/frames/padded_frames/<split>/                 — padded images + masks (generated)
#   data/frames/model_data/patches/<split>/            — training patches (generated)
#
# SPLIT drives the labelled pipeline (merge_csv → generate_masks → first_second →
# pad_images → split_patches → train); it needs a raw_csv_files/<split> folder with
# CSVs. PREDICT_SPLIT drives inference only (first_second → predict) and needs no CSV
# — just raw frames in data/frames/raw_frames/<PREDICT_SPLIT>/.
#
# Configure the variables below (or override on the command line, e.g.
# `make prepare_data SPLIT=val`) before running.
# =============================================================================

PYTHON ?= python3

# ─── Split selection ──────────────────────────────────────────────────────────
# NOTE: keep these assignments free of trailing inline comments — GNU Make bakes
# any whitespace before a same-line "#" into the variable's value.
SPLIT         ?= train
PREDICT_SPLIT ?= test

# ─── Paths (derived from SPLIT / PREDICT_SPLIT) ───────────────────────────────
RAW_CSV_DIR       := data/csv/raw_csv_files/$(SPLIT)
SPLIT_CSV         := data/csv/csv_per_split/$(SPLIT).csv

RAW_FRAMES_DIR         := data/frames/raw_frames/$(SPLIT)
FIRST_SECOND_DIR       := data/frames/first_second/$(SPLIT)
# generate_tracking_output.py appends a subfolder named after the CSV (i.e. SPLIT)
# onto whatever --output_dir it's given, so MASKS_PARENT_DIR is what gets passed in.
MASKS_PARENT_DIR       := data/frames/model_data/output
MASKS_DIR              := $(MASKS_PARENT_DIR)/$(SPLIT)
PADDED_DIR             := data/frames/padded_frames/$(SPLIT)
PATCHES_DIR            := data/frames/model_data/patches/$(SPLIT)

PREDICT_RAW_FRAMES_DIR   := data/frames/raw_frames/$(PREDICT_SPLIT)
PREDICT_FIRST_SECOND_DIR := data/frames/first_second/$(PREDICT_SPLIT)
PREDICT_OUTPUT_DIR       ?= data/predictions

# ─── Image / patch parameters ─────────────────────────────────────────────────
IMG_HEIGHT        := 1104
IMG_WIDTH         := 1376
IMG_CHANNELS      := 3
PATCH_SIZE        := 128
IMG_FORMAT        := jpg

# ─── Training config ──────────────────────────────────────────────────────────
CONFIG            := config.yaml

# ─── Prediction config ─────────────────────────────────────────────────────────
# train.py saves the best checkpoint as displ_xy_segm_best_model_<run_name>.pt in
# its config output_dir ("." by default). Auto-pick the most recently trained one
# so `make predict` works without typing the path — override it explicitly once
# you know which run you want, e.g. make predict MODEL_PATH=path/to/best_model.pt
MODEL_PATH        ?= $(shell ls -t displ_xy_segm_best_model_*.pt 2>/dev/null | head -n 1)

# =============================================================================

.PHONY: all prepare_data merge_csv generate_masks first_second pad_images \
        split_patches train predict_first_second predict clean help

help:
	@echo "Targets: prepare_data, merge_csv, generate_masks, first_second, pad_images,"
	@echo "         split_patches, train, predict_first_second, predict, all, clean"
	@echo "Key variables: SPLIT=$(SPLIT) PREDICT_SPLIT=$(PREDICT_SPLIT) MODEL_PATH=$(MODEL_PATH)"

all: prepare_data train

# ─── Full data preparation (runs steps 1-5 in order, for SPLIT) ───────────────
prepare_data: merge_csv generate_masks first_second pad_images split_patches

# ─── Step 1: Combine per-video tracking CSVs into one CSV for SPLIT ───────────
# Skips (without erroring) if raw_csv_files/<SPLIT> has no CSVs yet — lets
# `make prepare_data SPLIT=val` etc. no-op cleanly on splits you haven't filled in.
merge_csv:
	@echo "\n>>> Step 1/5: Merging tracking CSVs for split '$(SPLIT)'..."
	@mkdir -p $(dir $(SPLIT_CSV))
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
			--output_dir  $(MASKS_PARENT_DIR) \
			--img_height  $(IMG_HEIGHT) \
			--img_width   $(IMG_WIDTH) ; \
	else \
		echo "No merged CSV at $(SPLIT_CSV) — skipping." ; \
	fi

# ─── Step 3: Raw frames → first/second consecutive-pair folders ───────────────
# Skips if raw_frames/<SPLIT> has no frames yet.
first_second:
	@echo "\n>>> Step 3/5: Building first/second frame pairs for '$(SPLIT)'..."
	@if ls $(RAW_FRAMES_DIR)/*.$(IMG_FORMAT) >/dev/null 2>&1; then \
		mkdir -p $(FIRST_SECOND_DIR) ; \
		cp $(RAW_FRAMES_DIR)/*.$(IMG_FORMAT) $(FIRST_SECOND_DIR)/ ; \
		$(PYTHON) prepare_data/generate_first_second_folder.py $(FIRST_SECOND_DIR) ; \
	else \
		echo "No .$(IMG_FORMAT) frames found in $(RAW_FRAMES_DIR) — skipping." ; \
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

# ─── Step 5: Split padded images into patches ─────────────────────────────────
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

# ─── Train ─────────────────────────────────────────────────────────────────────
# Uses config.yaml for hyperparameters, but points train/val data at the patches
# just generated for SPLIT (overriding whatever train_images/train_masks say).
train:
	@echo "\n>>> Training model on '$(SPLIT)' patches..."
	$(PYTHON) train.py --config $(CONFIG) \
		--train_images $(PATCHES_DIR)/input \
		--train_masks  $(PATCHES_DIR)/output

# ─── Predict ────────────────────────────────────────────────────────────────────
# Skips if raw_frames/<PREDICT_SPLIT> has no frames yet.
predict_first_second:
	@echo "\n>>> Building first/second frame pairs for '$(PREDICT_SPLIT)'..."
	@if ls $(PREDICT_RAW_FRAMES_DIR)/*.$(IMG_FORMAT) >/dev/null 2>&1; then \
		mkdir -p $(PREDICT_FIRST_SECOND_DIR) ; \
		cp $(PREDICT_RAW_FRAMES_DIR)/*.$(IMG_FORMAT) $(PREDICT_FIRST_SECOND_DIR)/ ; \
		$(PYTHON) predict/generate_first_second_folder.py $(PREDICT_FIRST_SECOND_DIR) ; \
	else \
		echo "No .$(IMG_FORMAT) frames found in $(PREDICT_RAW_FRAMES_DIR) — skipping." ; \
	fi

predict: predict_first_second
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "No trained model found (no displ_xy_segm_best_model_*.pt here yet)." >&2; \
		echo "Run 'make train' first, or pass one explicitly: make predict MODEL_PATH=path/to/best_model.pt" >&2; \
		exit 1; \
	fi
	@echo "\n>>> Running prediction on '$(PREDICT_SPLIT)'..."
	@mkdir -p $(PREDICT_OUTPUT_DIR)
	$(PYTHON) predict/segment_with_our_nn_and_predict_xy_displ.py \
		$(PREDICT_FIRST_SECOND_DIR) \
		$(PREDICT_OUTPUT_DIR) \
		--model_path $(MODEL_PATH) \
		--img_height $(IMG_HEIGHT) \
		--img_width  $(IMG_WIDTH) \
		--patch_size $(PATCH_SIZE)

# ─── Clean generated data (keeps raw CSVs, raw frames, and model weights) ──────
clean:
	@echo "Removing generated data for split '$(SPLIT)' and predict split '$(PREDICT_SPLIT)'..."
	rm -rf $(SPLIT_CSV) $(FIRST_SECOND_DIR) $(MASKS_DIR) $(PADDED_DIR) $(PATCHES_DIR) \
	       $(PREDICT_FIRST_SECOND_DIR)
