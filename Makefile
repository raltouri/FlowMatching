.PHONY: verify features runs table figures runs-stage2 figures-stage2 analysis-stage2 runs-stage3 figures-stage3 all clean

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
	$(PYTHON) -m src.figures_stage1

# --- Stage 2: flow matching to class prototypes -------------------------

runs-stage2:
	$(PYTHON) -m src.train_fm

figures-stage2:
	$(PYTHON) -m src.figures_stage2

runs-stage3:
	$(PYTHON) -m src.train_fm3 --e2e
	$(PYTHON) -m src.train_fm3 --guided

figures-stage3:
	$(PYTHON) -m src.figures_stage3

# Geometry, prototype-identity check, and flip analysis.
analysis-stage2:
	$(PYTHON) -m src.analysis_stage2

all: features runs table figures

# Removes derived artefacts only; never touches data/.
clean:
	rm -rf features/* results/runs.csv results/ckpt/* results/curves/* \
	       figures/stage1/* figures/stage2/*
