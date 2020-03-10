{%- extends 'display_priority.tpl' -%}

{% set nb_title = nb.metadata.get('title', '') %}
{% set nb_date = nb.metadata.get('date', '') %}
{% set nb_filename = nb.metadata.get('filename', '') %}
{% set nb_filename_with_path = nb.metadata.get('filename_with_path','') %}
{% set indexPage = nb_filename.startswith('index') %}
{% set download_nb = nb.metadata.get('download_nb','') %}
{% set download_nb_path = nb.metadata.get('download_nb_path','') %}
{% if nb_filename.endswith('.rst') %}
{% set nb_filename = nb_filename[:-4] %}
{% endif %}

{%- block header %}
<!doctype html>
<html lang="en">
	<head>
		<!-- Global site tag (gtag.js) - Google Analytics -->
		<script async src="https://www.googletagmanager.com/gtag/js?id=UA-54984338-7"></script>
		<script>
		  window.dataLayer = window.dataLayer || [];
		  function gtag(){dataLayer.push(arguments);}
		  gtag('js', new Date());

		  gtag('config', 'UA-54984338-7');
		</script>

		<meta charset="utf-8">
{% if nb_filename == 'index' %}
		<title>Quantitative Economics with Python</title>
{% else %}
		<title>{{nb_title}} &ndash; Quantitative Economics with Python</title>
{% endif %}
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<meta name="author" content="Quantitative Economics with Python">
		<meta name="keywords" content="Python, QuantEcon, Quantitative Economics, Economics, Sloan, Alfred P. Sloan Foundation, Tom J. Sargent, John Stachurski">
		<meta name="description" content="This website presents a set of lectures on quantitative economic modeling, designed and written by Thomas J. Sargent and John Stachurski.">
		<meta name="twitter:card" content="summary">
		<meta name="twitter:site" content="@quantecon">
		<meta name="twitter:title" content="{{nb_title}}">
		<meta name="twitter:description" content="This website presents a set of lectures on quantitative economic modeling, designed and written by Thomas J. Sargent and John Stachurski.">
		<meta name="twitter:creator" content="@quantecon">
		<meta name="twitter:image" content="https://assets.quantecon.org/img/qe-twitter-logo.png">
		<meta property="og:title" content="{{nb_title}}" />
		<meta property="og:type" content="website" />
		<meta property="og:url" content="https://python.quantecon.org/{{nb_filename_with_path}}.html" />
		<meta property="og:image" content="https://assets.quantecon.org/img/qe-og-logo.png" />
		<meta property="og:description" content="This website presents a set of lectures on quantitative economic modeling, designed and written by Thomas J. Sargent and John Stachurski." />
		<meta property="og:site_name" content="Quantitative Economics with Python" />

		<link rel="stylesheet" href="/_static/css/python.css?v=1.0">
		<link rel="stylesheet" href="https://assets.quantecon.org/css/menubar-20191108.css">
		<link rel="icon" href="/_static/img/favicon.ico" type="image/x-icon" />

		<link href="https://fonts.googleapis.com/css?family=Droid+Serif|Source+Sans+Pro:400,700" rel="stylesheet">

		<script defer src="https://use.fontawesome.com/releases/v5.6.3/js/solid.js" integrity="sha384-F4BRNf3onawQt7LDHDJm/hwm3wBtbLIfGk1VSB/3nn3E+7Rox1YpYcKJMsmHBJIl" crossorigin="anonymous"></script>
		<script defer src="https://use.fontawesome.com/releases/v5.6.3/js/brands.js" integrity="sha384-VLgz+MgaFCnsFLiBwE3ItNouuqbWV2ZnIqfsA6QRHksEAQfgbcoaQ4PP0ZeS0zS5" crossorigin="anonymous"></script>
		<script defer src="https://use.fontawesome.com/releases/v5.6.3/js/fontawesome.js" integrity="sha384-treYPdjUrP4rW5q82SnECO7TPVAz4bpas16yuE9F5o7CeBn2YYw1yr5oC8s8Mf8t" crossorigin="anonymous"></script>

	</head>

	<body>

		<div class="qemb"> <!-- QuantEcon menubar -->

			<p class="qemb-logo"><a href="https://quantecon.org/" title="quantecon.org"><span class="show-for-sr">QuantEcon</span></a></p>
		
			<div class="qemb-menu">
		
				<ul class="qemb-groups">
					<li>
						<span>Lectures</span>
						<ul>
						<li><a href="https://python.quantecon.org/" title="Quantitative Economics with Python"><span>Quantitative Economics with Python</span></a></li>
						<li><a href="https://julia.quantecon.org/" title="Quantitative Economics with Julia"><span>Quantitative Economics with Julia</span></a></li>
						<li><a href="https://datascience.quantecon.org/" title="DataScience"><span>QuantEcon DataScience</span></a></li>
						<li><a href="http://cheatsheets.quantecon.org/" title="Cheatsheets"><span>Cheatsheets</span></a></li>
						</ul>
					</li>
					<li>
						<span>Code</span>
						<ul>
						<li><a href="https://quantecon.org/quantecon-py" title="QuantEcon.py"><span>QuantEcon.py</span></a></li>
						<li><a href="https://quantecon.org/quantecon-jl" title="QuantEcon.jl"><span>QuantEcon.jl</span></a></li>
						<li><a href="https://jupinx.quantecon.org/">Jupinx</a></li>
						</ul>
					</li>
					<li>
						<span>Notebooks</span>
						<ul>
						<li><a href="https://quantecon.org/notebooks" title="QuantEcon Notebook Library"><span>NB Library</span></a></li>
						<li><a href="http://notes.quantecon.org/" title="QE Notes"><span>QE Notes</span></a></li>
						</ul>
					</li>
					<li>
						<span>Community</span>
						<ul>
						<li><a href="http://blog.quantecon.org/" title="Blog"><span>Blog</span></a></li>
						<li><a href="http://discourse.quantecon.org/" title="Forum"><span>Forum</span></a></li>
						</ul>
					</li>
				</ul>
		
				<ul class="qemb-links">
					<li><a href="http://store.quantecon.org/" title="Store"><span class="show-for-sr">Store</span></a></li>
					<li><a href="https://github.com/QuantEcon/" title="Repository"><span class="show-for-sr">Repository</span></a></li>
					<li><a href="https://twitter.com/quantecon" title="Twitter"><span class="show-for-sr">Twitter</span></a></li>
				</ul>
		
			</div>
	
		</div>

		<div class="wrapper">

			<header class="header">

				<div class="branding">

					<p class="site-title"><a href="/">Quantitative Economics with Python</a></p>

					<p class="sr-only"><a href="#skip">Skip to content</a></p>

					<ul class="site-authors">
						<li><a href="http://www.tomsargent.com/">Thomas J. Sargent</a></li>
						<li><a href="http://johnstachurski.net/">John Stachurski</a></li>
					</ul>

				</div>

				<div class="header-tools">

					<div class="site-search">
					<script async src="https://cse.google.com/cse.js?cx=006559439261123061640:j0o7s27tvxo"></script>
					<div class="gcse-searchbox-only" data-resultsUrl="/search.html" enableAutoComplete="true"></div>
					<script>window.onload = function(){ document.getElementById('gsc-i-id1').placeholder = 'Search'; };</script>
					</div>

{% if indexPage or nb_filename == 'status' %}
					<div class="header-badge" id="coverage_badge"></div>
{% else %}
					<div class="header-badge" id="executability_status_badge"></div>
{% endif %}

				</div>

			</header>

			<div class="main">

				<div class="breadcrumbs">
					<ul>
						<li><a href="https://quantecon.org/">Org</a> •</li>
						<li><a href="/">Home</a> &raquo;</li>
{% if not nb_filename == 'index_toc' %}
						<li><a href="/index_toc.html">Table of Contents</a> &raquo;</li>
{% endif %}
						<li>{{nb_title}}</li>
					</ul>
				</div>

				<!--
				<div class="announcement">
					<p>The announcement...</p>
				</div>
				-->

				<div class="content">

					<div id="skip"></div>

					<div class="document">

{% if not indexPage %}
						<div class="lecture-options">
							<ul>
{% if download_nb == True %}
								<li><a href="/_downloads/pdf/{{nb_filename_with_path}}.pdf"><i class="fas fa-file-download"></i> Download PDF</a></li>
								<li><a href="/_downloads/ipynb/{{nb_filename_with_path}}.ipynb"><i class="fas fa-file-download"></i> Download Notebook</a></li>
{% endif %}
								<li><a target="_blank" href="https://colab.research.google.com/github/QuantEcon/quantecon-notebooks-python/blob/master/{{nb_filename_with_path}}.ipynb"><i class="fas fa-rocket"></i> Launch Notebook</a></li>
								<li><a target="_blank" href="https://github.com/QuantEcon/lecture-source-py/blob/master/source/rst/{{nb_filename_with_path}}.rst"><i class="fas fa-file-code"></i> View Source</a></li>
							</ul>
							<ul>
								<li><a href="/troubleshooting.html"><i class="fas fa-question-circle"></i> Troubleshooting</a></li>
								<li><a href="https://github.com/QuantEcon/lecture-source-py/issues"><i class="fas fa-flag"></i> Report issue</a></li>
							</ul>
						</div>
{% endif %}


{%- endblock header-%}

{% block codecell %}
{% set html_class = cell['metadata'].get('html-class', {}) %}
<div class="{{ html_class }} cell border-box-sizing code_cell rendered">
{{ super() }}
</div>
{%- endblock codecell %}

{% block input_group -%}
<div class="input">
{{ super() }}
</div>
{% endblock input_group %}

{% block output_group %}
<div class="output_wrapper">
<div class="output">
{{ super() }}
</div>
</div>
{% endblock output_group %}

{% block in_prompt -%}
<div class="prompt input_prompt">
	{%- if cell.execution_count is defined -%}
		In&nbsp;[{{ cell.execution_count|replace(None, "&nbsp;") }}]:
	{%- else -%}
		In&nbsp;[&nbsp;]:
	{%- endif -%}
</div>
{%- endblock in_prompt %}

{% block empty_in_prompt -%}
<div class="prompt input_prompt">
</div>
{%- endblock empty_in_prompt %}

{# 
  output_prompt doesn't do anything in HTML,
  because there is a prompt div in each output area (see output block)
#}
{% block output_prompt %}
{% endblock output_prompt %}

{% block input %}
<div class="inner_cell">
	<div class="input_area">
{{ cell.source | highlight_code(metadata=cell.metadata) }}
	</div>
</div>
{%- endblock input %}

{% block output_area_prompt %}
{%- if output.output_type == 'execute_result' -%}
	<div class="prompt output_prompt">
	{%- if cell.execution_count is defined -%}
		Out[{{ cell.execution_count|replace(None, "&nbsp;") }}]:
	{%- else -%}
		Out[&nbsp;]:
	{%- endif -%}
{%- else -%}
	<div class="prompt">
{%- endif -%}
	</div>
{% endblock output_area_prompt %}

{% block output %}
<div class="output_area">
{% if resources.global_content_filter.include_output_prompt %}
	{{ self.output_area_prompt() }}
{% endif %}
{{ super() }}
</div>
{% endblock output %}

{% block markdowncell scoped %}
{% set html_class = cell['metadata'].get('html-class', {}) %}
<div class="{{ html_class }} cell border-box-sizing text_cell rendered">
{%- if resources.global_content_filter.include_input_prompt-%}
	{{ self.empty_in_prompt() }}
{%- endif -%}
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
{{ cell.source  | markdown2html | strip_files_prefix }}
</div>
</div>
</div>
{%- endblock markdowncell %}

{% block unknowncell scoped %}
unknown type  {{ cell.type }}
{% endblock unknowncell %}

{% block execute_result -%}
{%- set extra_class="output_execute_result" -%}
{% block data_priority scoped %}
{{ super() }}
{% endblock data_priority %}
{%- set extra_class="" -%}
{%- endblock execute_result %}

{% block stream_stdout -%}
<div class="output_subarea output_stream output_stdout output_text">
<pre>
{{- output.text | ansi2html -}}
</pre>
</div>
{%- endblock stream_stdout %}

{% block stream_stderr -%}
<div class="output_subarea output_stream output_stderr output_text">
<pre>
{{- output.text | ansi2html -}}
</pre>
</div>
{%- endblock stream_stderr %}

{% block data_svg scoped -%}
<div class="output_svg output_subarea {{ extra_class }}">
{%- if output.svg_filename %}
<img src="{{ output.svg_filename | posix_path }}"
{%- else %}
{{ output.data['image/svg+xml'] }}
{%- endif %}
</div>
{%- endblock data_svg %}

{% block data_html scoped -%}
<div class="output_html rendered_html output_subarea {{ extra_class }}">
{{ output.data['text/html'] }}
</div>
{%- endblock data_html %}

{% block data_markdown scoped -%}
<div class="output_markdown rendered_html output_subarea {{ extra_class }}">
{{ output.data['text/markdown'] | markdown2html }}
</div>
{%- endblock data_markdown %}

{% block data_png scoped %}
<div class="output_png output_subarea {{ extra_class }}">
{%- if 'image/png' in output.metadata.get('filenames', {}) %}
<img src="{{ output.metadata.filenames['image/png'] | posix_path }}"
{%- else %}
<img src="data:image/png;base64,{{ output.data['image/png'] }}"
{%- endif %}
{%- set width=output | get_metadata('width', 'image/png') -%}
{%- if width is not none %}
width={{ width }}
{%- endif %}
{%- set height=output | get_metadata('height', 'image/png') -%}
{%- if height is not none %}
height={{ height }}
{%- endif %}
{%- if output | get_metadata('unconfined', 'image/png') %}
class="unconfined"
{%- endif %}
>
</div>
{%- endblock data_png %}

{% block data_jpg scoped %}
<div class="output_jpeg output_subarea {{ extra_class }}">
{%- if 'image/jpeg' in output.metadata.get('filenames', {}) %}
<img src="{{ output.metadata.filenames['image/jpeg'] | posix_path }}"
{%- else %}
<img src="data:image/jpeg;base64,{{ output.data['image/jpeg'] }}"
{%- endif %}
{%- set width=output | get_metadata('width', 'image/jpeg') -%}
{%- if width is not none %}
width={{ width }}
{%- endif %}
{%- set height=output | get_metadata('height', 'image/jpeg') -%}
{%- if height is not none %}
height={{ height }}
{%- endif %}
{%- if output | get_metadata('unconfined', 'image/jpeg') %}
class="unconfined"
{%- endif %}
>
</div>
{%- endblock data_jpg %}

{% block data_latex scoped %}
<div class="output_latex output_subarea {{ extra_class }}">
{{ output.data['text/latex'] }}
</div>
{%- endblock data_latex %}

{% block error -%}
<div class="output_subarea output_text output_error">
<pre>
{{- super() -}}
</pre>
</div>
{%- endblock error %}

{%- block traceback_line %}
{{ line | ansi2html }}
{%- endblock traceback_line %}

{%- block data_text scoped %}
<div class="output_text output_subarea {{ extra_class }}">
<pre>
{{- output.data['text/plain'] | ansi2html -}}
</pre>
</div>
{%- endblock -%}

{%- block data_javascript scoped %}
{% set div_id = uuid4() %}
<div id="{{ div_id }}"></div>
<div class="output_subarea output_javascript {{ extra_class }}">
<script type="text/javascript">
var element = $('#{{ div_id }}');
{{ output.data['application/javascript'] }}
</script>
</div>
{%- endblock -%}

{%- block data_widget_state scoped %}
{% set div_id = uuid4() %}
{% set datatype_list = output.data | filter_data_type %} 
{% set datatype = datatype_list[0]%} 
<div id="{{ div_id }}"></div>
<div class="output_subarea output_widget_state {{ extra_class }}">
<script type="text/javascript">
var element = $('#{{ div_id }}');
</script>
<script type="{{ datatype }}">
{{ output.data[datatype] | json_dumps }}
</script>
</div>
{%- endblock data_widget_state -%}

{%- block data_widget_view scoped %}
{% set div_id = uuid4() %}
{% set datatype_list = output.data | filter_data_type %} 
{% set datatype = datatype_list[0]%} 
<div id="{{ div_id }}"></div>
<div class="output_subarea output_widget_view {{ extra_class }}">
<script type="text/javascript">
var element = $('#{{ div_id }}');
</script>
<script type="{{ datatype }}">
{{ output.data[datatype] | json_dumps }}
</script>
</div>
{%- endblock data_widget_view -%}

{%- block footer %}
{% set mimetype = 'application/vnd.jupyter.widget-state+json'%} 
{% if mimetype in nb.metadata.get("widgets",{})%}
<script type="{{ mimetype }}">
{{ nb.metadata.widgets[mimetype] | json_dumps }}
</script>
{% endif %}
{{ super() }}


					</div>

				</div>

			</div>

			<footer class="footer">

				<p class="logo"><a href="#"><img src="/_static/img/qe-logo.png"></a></p>

				<p><a rel="license" href="https://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" src="https://i.creativecommons.org/l/by-sa/4.0/80x15.png" /></a></p>

				<p>This work is licensed under a <a rel="license" href="https://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International</a>.</p>

				<p>&copy; Copyright 2020, Thomas J. Sargent and John Stachurski. Created using <a href="https://jupinx.quantecon.org/">Jupinx</a>, hosted with <a href="https://aws.amazon.com/">AWS</a>.</p>

			</footer>

		</div>

		<div class="page-tools">

			<ul>
				<li class="top"><a href="#top" title="Back to top"><i class="fas fa-chevron-up"></i></a></li>
				<li><a href="http://twitter.com/intent/tweet?url=https%3A%2F%2Fpython.quantecon.org%2F{{nb_filename_with_path}}.html&via=QuantEcon&text={{nb_title}}" title="Share on Twitter" target="_blank"><i class="fab fa-twitter"></i></a></li>
				<li><a href="https://www.linkedin.com/shareArticle?mini=true&url=https://python.quentecon.org%2F{{nb_filename_with_path}}.html&title={{nb_title}}&summary=This%20website%20presents%20a%20series%20of%20lectures%20on%20quantitative%20economic%20modeling,%20designed%20and%20written%20by%20Thomas%20J.%20Sargent%20and%20John%20Stachurski.&source=QuantEcon" title="Share on LinkedIn" target="_blank"><i class="fab fa-linkedin-in"></i></a></li>
				<li><a href="https://www.facebook.com/sharer/sharer.php?u=https%3A//python.quantecon.org%2F{{nb_filename_with_path}}.html" title="Share on Facebook" target="_blank"><i class="fab fa-facebook-f"></i></a></li>
				<li><span class="title">Share page</span></li>
			</ul>

		</div>

		<div id="nb_date" style="display:none;">{{nb_date}}</div>

		<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
		<script src="https://assets.quantecon.org/js/menubar-20191106.js"></script>
		<script src="/_static/js/python.js?v=1.0"></script>

	</body>
</html>


{%- endblock footer-%}
