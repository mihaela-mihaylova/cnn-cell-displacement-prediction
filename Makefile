# =============================================================================
# Cell Displacement Prediction Pipeline
# =============================================================================
# Usage:
#   make prepare_data   — run all 3 data preparation steps in order
#   make train          — train the model
#   make predict        — run prediction on new data
#   make all            — prepare data + train
#
# Configure the variables below before running.
# =============================================================================

# ─── Paths ────────────────────────────────────────────────────────────────────
TRACKING_CSV      := data/tracking.csv
RAW_FRAMES_DIR    := data/raw_frames

PADDED_DIR        := data/padded
MASKS_DIR         := data/masks
PATCHES_DIR       := data/patches

PREDICT_INPUT     := data/predict_input
PREDICT_OUTPUT    := data/predict_output

# ─── Image parameters ─────────────────────────────────────────────────────────
IMG_HEIGHT        := 1104
IMG_WIDTH         := 1376
IMG_CHANNELS      := 3
PATCH_SIZE        := 128

# ─── Tracking parameters ──────────────────────────────────────────────────────
FIRST_FRAME_ID    := 0
# Coordinate scaling: 1.0 = CSV matches image size,
# 0.5 = CSV was extracted from images twice as large
COEFF             := 1.0

# ─── Training config ──────────────────────────────────────────────────────────
CONFIG            := config.yaml

# ─── Prediction config ────────────────────────────────────────────────────────
# Set this to the path of your trained model .pt file before running make predict
MODEL_PATH        := outputs/displ_xy_segm_best_model.pt

# =============================================================================

.PHONY: all prepare_data generate_masks pad_images split_patches train predict clean

all: prepare_data train

# ─── Full data preparation (runs steps 1, 2, 3 in order) ─────────────────────
prepare_data: generate_masks pad_images split_patches

# ─── Step 1: CSV → position and displacement masks ───────────────────────────
generate_masks:
	@echo "\n>>> Step 1/3: Generating position and displacement masks..."
	python 00_prepare_data/01_generate_tracking_output.py \
		--csv            $(TRACKING_CSV) \
		--output_dir     $(MASKS_DIR) \
		--img_height     $(IMG_HEIGHT) \
		--img_width      $(IMG_WIDTH) \
		--first_frame_id $(FIRST_FRAME_ID) \
		--coeff          $(COEFF)

# ─── Step 2: Pad images and masks ─────────────────────────────────────────────
# Padding is calculated automatically from img_height, img_width, patch_size.
# Run once per folder: first/, second/, and all three mask types.
pad_images:
	@echo "\n>>> Step 2/3: Padding images and masks..."
	python 00_prepare_data/02_pad_images.py \
		--input_dir    $(RAW_FRAMES_DIR)/first \
		--output_dir   $(PADDED_DIR)/first \
		--img_height   $(IMG_HEIGHT) \
		--img_width    $(IMG_WIDTH) \
		--patch_size   $(PATCH_SIZE) \
		--img_channels $(IMG_CHANNELS)
	python 00_prepare_data/02_pad_images.py \
		--input_dir    $(RAW_FRAMES_DIR)/second \
		--output_dir   $(PADDED_DIR)/second \
		--img_height   $(IMG_HEIGHT) \
		--img_width    $(IMG_WIDTH) \
		--patch_size   $(PATCH_SIZE) \
		--img_channels $(IMG_CHANNELS)
	python 00_prepare_data/02_pad_images.py \
		--input_dir    $(MASKS_DIR)/pos_masks \
		--output_dir   $(PADDED_DIR)/pos_masks \
		--img_height   $(IMG_HEIGHT) \
		--img_width    $(IMG_WIDTH) \
		--patch_size   $(PATCH_SIZE) \
		--img_channels 1 \
		--format       png
	python 00_prepare_data/02_pad_images.py \
		--input_dir    $(MASKS_DIR)/displ_masks/displ_x \
		--output_dir   $(PADDED_DIR)/displ_masks/displ_x \
		--img_height   $(IMG_HEIGHT) \
		--img_width    $(IMG_WIDTH) \
		--patch_size   $(PATCH_SIZE) \
		--img_channels $(IMG_CHANNELS) \
		--format       png
	python 00_prepare_data/02_pad_images.py \
		--input_dir    $(MASKS_DIR)/displ_masks/displ_y \
		--output_dir   $(PADDED_DIR)/displ_masks/displ_y \
		--img_height   $(IMG_HEIGHT) \
		--img_width    $(IMG_WIDTH) \
		--patch_size   $(PATCH_SIZE) \
		--img_channels $(IMG_CHANNELS) \
		--format       png

# ─── Step 3: Split padded images into patches ─────────────────────────────────
split_patches:
	@echo "\n>>> Step 3/3: Splitting into patches..."
	python 00_prepare_data/03_split_images_to_patches.py \
		--input_dir    $(PADDED_DIR) \
		--output_dir   $(PATCHES_DIR) \
		--img_height   $(IMG_HEIGHT) \
		--img_width    $(IMG_WIDTH) \
		--patch_size   $(PATCH_SIZE) \
		--img_channels $(IMG_CHANNELS)

# ─── Train ────────────────────────────────────────────────────────────────────
train:
	@echo "\n>>> Training model..."
	python 01_train/train.py --config $(CONFIG)

# ─── Predict ──────────────────────────────────────────────────────────────────
predict:
	@echo "\n>>> Running prediction..."
	python 02_predict/generate_first_second_folder.py $(PREDICT_INPUT)
	python 02_predict/segment_with_our_nn_and_predict_xy_displ.py \
		$(PREDICT_INPUT) \
		$(PREDICT_OUTPUT) \
		--model_path $(MODEL_PATH) \
		--img_height $(IMG_HEIGHT) \
		--img_width  $(IMG_WIDTH) \
		--patch_size $(PATCH_SIZE)

# ─── Clean generated data (keeps raw data and model weights) ──────────────────
clean:
	@echo "Removing generated data..."
	rm -rf $(PADDED_DIR) $(PATCHES_DIR) $(MASKS_DIR)