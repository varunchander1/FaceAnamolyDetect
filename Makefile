VLM_MODEL ?= gpt-5-mini
VLM_FALLBACK ?= gpt-5-mini

FACES_DIR ?= faces
GALLERY ?= gallery.npz
EXPORT_DIR ?= export

.PHONY: run seed gallery
run:
	VLM_MODEL=$(VLM_MODEL) VLM_FALLBACK=$(VLM_FALLBACK) \
		uv run streamlit run app.py --server.runOnSave=false

seed:
	uv run python tools/seed_faces.py --export-dir $(EXPORT_DIR) --faces-dir $(FACES_DIR)

gallery:
	uv run python tools/build_gallery.py --faces-dir $(FACES_DIR) --out $(GALLERY)

labelstudio:
	docker compose up -d