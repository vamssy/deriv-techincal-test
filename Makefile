.PHONY: run validate clean

# Regenerate all artifacts from disk (mock provider by default; NVIDIA if NVIDIA_API_KEY is set)
run:
	python -m src.run

# Assert the full grading surface (forces the deterministic mock provider)
validate:
	python validate.py

# Remove all generated artifacts
clean:
	rm -rf out
