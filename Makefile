SHELL := bash
#
# Makefile for Sphinx Extension Test Cases
#

# You can set these variables from the command line.
SPHINXOPTS    = -c "./"
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = lecture-source-py
SOURCEDIR     = source/rst
BUILDDIR      = _build
BUILDWEBSITE  = _build/website
BUILDCOVERAGE = _build/coverage
BUILDPDF      = _build/pdf
PORT          = 8890
FILES         = 

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(FILES) $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Install requiremenets for building lectures.
setup:
	pip install -r requirements.txt

preview:
ifneq (,$(filter $(target),website Website))
	cd $(BUILDDIR)/jupyter_html && python -m http.server $(PORT)
else
ifdef lecture
	cd $(BUILDDIR)/jupyter/ && jupyter notebook --port $(PORT) --port-retries=0 $(basename $(lecture)).ipynb
else
	cd $(BUILDDIR)/jupyter/ && jupyter notebook --port $(PORT) --port-retries=0
endif
endif

clean-coverage:
	rm -rf $(BUILDCOVERAGE)

clean-website:
	rm -rf $(BUILDDIR)/jupyterhtml
	rm -rf $(BUILDDIR)/jupyter_html

clean-pdf:
	rm -rf $(BUILDDIR)/jupyterpdf

clean-execute:
	rm -rf $(BUILDDIR)/execute

clean-jupyter:
	rm -rf $(BUILDDIR)/jupyter
 
website:
	@$(SPHINXBUILD) -M jupyterhtml "$(SOURCEDIR)" "$(BUILDDIR)" $(FILES) $(SPHINXOPTS) $(O) -D jupyter_download_nb_image_urlpath="https://s3-ap-southeast-2.amazonaws.com/python.quantecon.org/_static/" -D jupyter_images_markdown=0 -D jupyter_html_template="theme/templates/python-html.tpl" -D jupyter_download_nb_urlpath="https://python.quantecon.org/"

pdf:
	@$(SPHINXBUILD) -M jupyterpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(FILES) $(SPHINXOPTS) $(O) -D jupyter_latex_template="theme/templates/latex.tpl" -D jupyter_latex_template_book="theme/templates/latex_book.tpl" -D jupyter_images_markdown=1 -D jupyter_target_pdf=1

execute:
ifneq ($(strip $(parallel)),)
	@@$(SPHINXBUILD) -M execute "$(SOURCEDIR)" "$(BUILDDIR)" $(FILES) $(SPHINXOPTS) $(O) -D jupyter_template_coverage_file_path="theme/templates/error_report_template.html" -D jupyter_number_workers=$(parallel)

else
	@$(SPHINXBUILD) -M execute "$(SOURCEDIR)" "$(BUILDDIR)" $(FILES) $(SPHINXOPTS) $(O) -D jupyter_template_coverage_file_path="theme/templates/error_report_template.html"
endif

# constructor-pdf:
# ifneq ($(strip $(parallel)),)
# 	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDPDF)" $(FILES) $(SPHINXOPTS) $(O) -D jupyter_images_markdown=1 -D jupyter_execute_notebooks=1 -D jupyter_number_workers=$(parallel)

# else
# 	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDPDF)" $(FILES) $(SPHINXOPTS) $(O) -D jupyter_images_markdown=1 -D jupyter_execute_notebooks=1
# endif

notebooks:
	make jupyter

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(FILES) $(SPHINXOPTS) $(O) -D jupyter_allow_html_only=1
