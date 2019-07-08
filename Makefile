# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = lecture-source-jl
SOURCEDIR     = source/rst
BUILDDIR      = _build
BUILDWEBSITE  = _build_website
CORES 		  = 4
BUILDCOVERAGE = _build_coverage

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Install requiremenets for building lectures.
setup:
	pip install -r requirements.txt

preview:
ifneq (,$(filter $(target),website Website))
	cd _build/jupyter_html/ && python -m http.server
else
ifdef lecture
	cd _build/jupyter/ && jupyter notebook $(basename $(lecture)).ipynb
else
	cd _build/jupyter/ && jupyter notebook
endif
endif

clean-coverage:
	rm -rf $(BUILDCOVERAGE)

clean-website:
	rm -rf $(BUILDWEBSITE)

coverage:
ifneq (,$(filter $(parallel),true True))
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDCOVERAGE)" $(SPHINXOPTS) $(O) -D jupyter_make_coverage=1 -D jupyter_execute_notebooks=1 -D jupyter_ignore_skip_test=0 -D jupyter_template_coverage_file_path="theme/templates/error_report_template.html" -D jupyter_number_workers=$(CORES) 
else
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDCOVERAGE)" $(SPHINXOPTS) $(O) -D jupyter_make_coverage=1 -D jupyter_execute_notebooks=1 -D jupyter_ignore_skip_test=0 -D jupyter_template_coverage_file_path="theme/templates/error_report_template.html"
endif

website:
ifneq (,$(filter $(parallel),true True))
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDWEBSITE)" $(SPHINXOPTS) $(O) -D jupyter_make_site=1 -D jupyter_generate_html=1 -D jupyter_download_nb=1 -D jupyter_execute_notebooks=1 -D jupyter_target_html=1 -D jupyter_images_urlpath="https://s3-ap-southeast-2.amazonaws.com/lectures.quantecon.org/py/_static/" -D jupyter_images_markdown=0 -D jupyter_html_template="theme/templates/lectures-nbconvert.tpl" -D jupyter_download_nb_urlpath="https://lectures.quantecon.org/" -D jupyter_number_workers=$(CORES)

else
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDWEBSITE)" $(SPHINXOPTS) $(O) -D jupyter_make_site=1 -D jupyter_generate_html=1 -D jupyter_download_nb=1 -D jupyter_execute_notebooks=1 -D jupyter_target_html=1 -D jupyter_images_urlpath="https://s3-ap-southeast-2.amazonaws.com/lectures.quantecon.org/py/_static/" -D jupyter_images_markdown=0 -D jupyter_html_template="theme/templates/lectures-nbconvert.tpl" -D jupyter_download_nb_urlpath="https://lectures.quantecon.org/"
endif

notebooks:
	make jupyter

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -D jupyter_allow_html_only=1