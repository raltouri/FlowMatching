.PHONY: verify features runs figures all clean

# M0 gate: confirm the official splits load with the expected counts.
verify:
	python -m src.data --verify

# M2: build the three feature caches (images are read exactly once).
features:
	python -m src.extract

# M3 + M4: the 21 prototype evaluations and the 27 linear-probe runs.
runs:
	python -m src.train

# M6: regenerate every figure from results/runs.csv and the caches.
figures:
	python -m src.figures

all: features runs figures

# Removes derived artefacts only; never touches data/.
clean:
	rm -rf features/* results/runs.csv results/ckpt/* figures/*
