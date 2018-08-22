# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = QuantEconlectures
SOURCEDIR     = .
BUILDDIR      = _build

#Check for make.lock file. If not equal then set BUILD = True
ifeq ("$(wildcard ./make.lock)","") 
BUILD=TRUE
else
BUILD=FALSE
endif

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

check_environment:
ifeq ($(BUILD),FALSE) 
	@echo "[ERROR] You cannot run make in the source directory"; exit 1 
else
	@true
endif

setup: check_environment install_dependancies
	python build.py

install_dependancies: check_environment
	pip install --upgrade sphinxcontrib-tikz        
	pip install --upgrade sphinxcontrib-bibtex 		

web: check_environment
	make html

# force jupyter to build using jupyter_conversion_mode="all"
jupyter-all: check_environment
	sphinx-build -D jupyter_conversion_mode="all" -b jupyter "$(SOURCEDIR)" "$(BUILDDIR)/jupyter"

xelatexpdf: latex
	@echo "Running LaTeX files through xelatex to construct PDF with Unicode..."
	$(MAKE) PDFLATEX=xelatex -C $(BUILDDIR)/latex all-pdf
	$(MAKE) PDFLATEX=xelatex -C $(BUILDDIR)/latex all-pdf
	@echo "xelatex finished; the PDF files are in $(BUILDDIR)/latex."


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile check_environment
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)