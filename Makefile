.PHONY: verify features runs table figures all clean

PYTHON := .venv/bin/python

# M0 gate: confirm the official splits load with the expected counts.
verify:
	$(PYTHON) -m src.data --verify

# M2: build the three feature caches (images are read exactly once).
features:
	$(PYTHON) -m src.extract

# M3 + M4: the 21 prototype evaluations and the 27 linear-probe runs.
runs:
	$(PYTHON) -m src.train

# M5: aggregate the ledger into the accuracy table.
table:
	$(PYTHON) -m src.evaluate

# M6: regenerate every figure from results/runs.csv and the caches.
figures:
	$(PYTHON) -m src.figures

all: features runs table figures

# Removes derived artefacts only; never touches data/.
clean:
	rm -rf features/* results/runs.csv results/ckpt/* figures/*
