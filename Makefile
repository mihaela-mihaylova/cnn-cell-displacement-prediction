# =============================================================================
# Cell Displacement Prediction Pipeline
# =============================================================================
# Usage:
#   make prepare_data   — run all data-preparation steps for SPLIT, in order
#   make train          — train the model on SPLIT's patches
#   make all            — prepare data + train
#   make clean          — remove generated (derived) data for SPLIT
#
# Data layout expected under data/ (already scaffolded in this repo):
#   data/raw/raw_csv_files/<split>/*.csv                    — one CSV per tracked video (you provide)
#   data/raw/raw_frames/<split>/<video>/*.jpg               — one subfolder per video, each locally
#                                                              0-indexed (0.jpg, 1.jpg, ...); subfolder
#                                                              names must share the matching CSV's leading
#                                                              "<n>_" prefix (e.g. "1_...") so videos line
#                                                              up across steps (you provide)
#   data/intermediate/<split>.csv             — merged CSV (generated)
#   data/intermediate/<split>/first_second//{first,second}/  — consecutive frame pairs (generated)
#   data/intermediate/<split>/masks/                        — position + displacement masks (generated)
#   data/intermediate/<split>/padded/                       — padded images + masks (generated)
#   data/model_data/<split>/input/{first,second}            — final training patches (generated)
#   data/model_data/<split>/output/{pos_masks,displ_masks}  — final training patches (generated)
#
# SPLIT drives the labelled pipeline (merge_csv → generate_masks → first_second →
# pad_images → split_patches → train); it needs a raw_csv_files/<split> folder with CSVs.
#
# Configure the variables below (or override on the command line, e.g.
# `make prepare_data SPLIT=val`) before running.
# =============================================================================

PYTHON ?= python3

# ─── Split selection ──────────────────────────────────────────────────────────
# NOTE: keep these assignments free of trailing inline comments — GNU Make bakes
# any whitespace before a same-line "#" into the variable's value.
SPLIT         ?= train
ALL_SPLITS    ?= train val test

# ─── Paths (derived from SPLIT) ────────────────────────────────
RAW_DIR           := data/raw
RAW_CSV_DIR       := $(RAW_DIR)/raw_csv_files/$(SPLIT)
RAW_FRAMES_DIR    := $(RAW_DIR)/raw_frames/$(SPLIT)

INTERMEDIATE_DIR  := data/intermediate
SPLIT_CSV         := $(INTERMEDIATE_DIR)/$(SPLIT).csv
FIRST_SECOND_DIR  := $(INTERMEDIATE_DIR)/$(SPLIT)/first_second
MASKS_PARENT_DIR  := $(INTERMEDIATE_DIR)/$(SPLIT)/masks
MASKS_DIR         := $(MASKS_PARENT_DIR)
PADDED_DIR        := $(INTERMEDIATE_DIR)/$(SPLIT)/padded

# Final patches: split_images_to_patches.py always writes <output_dir>/input and
# <output_dir>/output itself, so split_patches stages into a scratch dir per SPLIT
# and moves the two halves into data/model_data/{input,output}/<split>.
MODEL_DATA_DIR      := data/model_data
PATCHES_INPUT_DIR   := $(MODEL_DATA_DIR)/$(SPLIT)/input
PATCHES_OUTPUT_DIR  := $(MODEL_DATA_DIR)/$(SPLIT)/output/
PATCHES_STAGE_DIR   := $(MODEL_DATA_DIR)/.staging_$(SPLIT)

# ─── Image / patch parameters ─────────────────────────────────────────────────
IMG_HEIGHT        := 1104
IMG_WIDTH         := 1376
IMG_CHANNELS      := 3
PATCH_SIZE        := 128
IMG_FORMAT        := jpg

# ─── Training config ──────────────────────────────────────────────────────────
CONFIG            := config.yaml
# =============================================================================

.PHONY: all prepare_data prepare_data_all merge_csv generate_masks first_second \
        first_second_all pad_images split_patches train \
        clean help

help:
	@echo "Targets: prepare_data, prepare_data_all, merge_csv, generate_masks, first_second,"
	@echo "         first_second_all, pad_images, split_patches, train"
	@echo "         all, clean"
	@echo "Key variables: SPLIT=$(SPLIT) ALL_SPLITS=$(ALL_SPLITS)"

all: prepare_data train

# ─── Full data preparation (runs steps 1-5 in order, for SPLIT) ───────────────
prepare_data: merge_csv generate_masks first_second pad_images split_patches

# Each step still skips cleanly per split if that split has no raw data yet -- e.g.
# if you haven't populated a given split's raw_csv_files/ or raw_frames/ yet.
prepare_data_all:
	@for s in $(ALL_SPLITS); do \
		$(MAKE) --no-print-directory prepare_data SPLIT=$$s ; \
	done

# ─── Just the first/second step, looped over ALL_SPLITS ──────────────────────
first_second_all:
	@for s in $(ALL_SPLITS); do \
		$(MAKE) --no-print-directory first_second SPLIT=$$s ; \
	done

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
			--img_width   $(IMG_WIDTH) \
			--for_training ; \
	else \
		echo "No merged CSV at $(SPLIT_CSV) — skipping." ; \
	fi

# ─── Step 3: Raw frames → first/second consecutive-pair folders ───────────────
# Skips if raw_frames/<SPLIT> has no frames yet.
first_second:
	@echo "\n>>> Step 3/5: Building first/second frame pairs for '$(SPLIT)'..."
		@if find $(RAW_FRAMES_DIR) -mindepth 1 -maxdepth 1 -type d 2>/dev/null | grep -q .; then \
		$(PYTHON) prepare_data/generate_first_second_folder.py \
			$(RAW_FRAMES_DIR) $(FIRST_SECOND_DIR) True ; \
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
# Skips if Step 4 produced no padded frames for SPLIT. Writes into a scratch
# staging dir (since the script always names its own input/output subfolders),
# then moves those into data/model_data/{input,output}/<split>.
split_patches:
	@echo "\n>>> Step 5/5: Splitting into patches..."
	@if [ -d $(PADDED_DIR)/padded_first ]; then \
		rm -rf $(PATCHES_STAGE_DIR) ; \
		$(PYTHON) prepare_data/split_images_to_patches.py \
			--input_dir    $(PADDED_DIR) \
			--output_dir   $(PATCHES_STAGE_DIR) \
			--img_height   $(IMG_HEIGHT) \
			--img_width    $(IMG_WIDTH) \
			--patch_size   $(PATCH_SIZE) \
			--img_channels $(IMG_CHANNELS) ; \
		mkdir -p $(MODEL_DATA_DIR)/input $(MODEL_DATA_DIR)/output ; \
		rm -rf $(PATCHES_INPUT_DIR) $(PATCHES_OUTPUT_DIR) ; \
		mv $(PATCHES_STAGE_DIR)/input  $(PATCHES_INPUT_DIR) ; \
		mv $(PATCHES_STAGE_DIR)/output $(PATCHES_OUTPUT_DIR) ; \
		mv $(PATCHES_STAGE_DIR)/patch_record.csv $(MODEL_DATA_DIR)/patch_record_$(SPLIT).csv ; \
		rm -rf $(PATCHES_STAGE_DIR) ; \
	else \
		echo "No padded frames found for '$(SPLIT)' — skipping patch splitting." ; \
	fi

# ─── Train ─────────────────────────────────────────────────────────────────────
# Uses config.yaml for hyperparameters, but points train/val data at the patches
# just generated for SPLIT (overriding whatever train_images/train_masks say).
train:
	@echo "\n>>> Training model on '$(SPLIT)' patches..."
	$(PYTHON) train.py --config $(CONFIG) \
		--train_images $(PATCHES_INPUT_DIR) \
		--train_masks  $(PATCHES_OUTPUT_DIR)

# ─── Clean generated data (keeps raw CSVs, raw frames, and model weights) ──────
clean:
	@echo "Removing generated data for split '$(SPLIT)'..."
	rm -rf $(SPLIT_CSV) $(FIRST_SECOND_DIR) $(MASKS_DIR) $(PADDED_DIR) \
	       $(PATCHES_INPUT_DIR) $(PATCHES_OUTPUT_DIR) $(PATCHES_STAGE_DIR) \
	       $(MODEL_DATA_DIR)/patch_record_$(SPLIT).csv