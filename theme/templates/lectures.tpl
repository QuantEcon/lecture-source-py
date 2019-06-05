{%- extends 'display_priority.tpl' -%}

{%- block header %}
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>Covariance Stationary Processes &ndash; Quantitative Economics</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
		<meta name="author" content="Quantitative Economics">
		<meta name="keywords" content="QuantEcon, Quantitative Economics, Economics, Sloan, Alfred P. Sloan Foundation, Tom J. Sargent, John Stachurski">
		<meta name="description" content="This website presents a series of lectures on quantitative economic modeling, designed and written by Thomas J. Sargent and John Stachurski.">
		<meta name="twitter:card" content="summary">
		<meta name="twitter:site" content="@quantecon">
		<meta name="twitter:title" content="Covariance Stationary Processes">
		<meta name="twitter:description" content="This website presents a series of lectures on quantitative economic modeling, designed and written by Thomas J. Sargent and John Stachurski.">
		<meta name="twitter:creator" content="@quantecon">
		<meta name="twitter:image" content="https://lectures.quantecon.org/_static/img/qeco-logo.png">
		<meta property="og:title" content="Covariance Stationary Processes" />
		<meta property="og:type" content="website" />
		<meta property="og:url" content="https://lectures.quantecon.org/py/arma.html" />
		<meta property="og:image" content="https://lectures.quantecon.org/_static/img/qeco-logo.png" />
		<meta property="og:description" content="This website presents a series of lectures on quantitative economic modeling, designed and written by Thomas J. Sargent and John Stachurski." />
		<meta property="og:site_name" content="Quantitative Economics" />

        <link rel="stylesheet" href="https://lectures.quantecon.org/_static/css/basic.css">
        <link rel="stylesheet" href="https://lectures.quantecon.org/_static/css/qe.css?v=1.2">
        <link rel="stylesheet" href="https://lectures.quantecon.org/_static/css/qe-menubar.css">
        <link rel="icon" href="https://lectures.quantecon.org/_static/img/favicon.ico" type="image/x-icon" />
		<link rel="canonical" href="https://lectures.quantecon.org/py/arma.html">

        <link href="https://fonts.googleapis.com/css?family=Droid+Serif|Source+Sans+Pro:400,700" rel="stylesheet">

    </head>


    <body>

        <div class="qe-menubar">
        
            <p class="qe-menubar-logo"><a href="https://quantecon.org/" title="quantecon.org"><img src="https://lectures.quantecon.org/_static/img/qe-menubar-logo.png" alt="QuantEcon logo"> Quant<span>Econ</span></a></p>
        
            <ul class="qe-menubar-nav">
                <li><a href="https://lectures.quantecon.org/" title="Lectures"><span>Lectures</span></a></li>
                <li><a href="https://quantecon.org/quantecon-py" title="QuantEcon.py"><span>QuantEcon.py</span></a></li>
                <li><a href="https://quantecon.org/quantecon-jl" title="QuantEcon.jl"><span>QuantEcon.jl</span></a></li>
                <li><a href="http://notes.quantecon.org/" title="QE Notes"><span>QE Notes</span></a></li>
                <li><a href="http://cheatsheets.quantecon.org/" title="Cheatsheets"><span>Cheatsheets</span></a></li>
                <li><a href="http://blog.quantecon.org/" title="Blog"><span>Blog</span></a></li>
                <li><a href="http://discourse.quantecon.org/" title="Forum"><span>Forum</span></a></li>
                <li><a href="http://store.quantecon.org/" title="Store"><span class="show-for-sr">Store</span></a></li>
                <li><a href="https://github.com/QuantEcon/" title="Repository"><span class="show-for-sr">Repository</span></a></li>
                <li><a href="https://twitter.com/quantecon" title="Twitter"><span class="show-for-sr">Twitter</span></a></li>
            </ul>
            
        </div>

        <div class="wrapper">

	        <header class="header">

				<div class="container">

		        	<div class="branding">

			        	<p class="site-title"><a href="https://lectures.quantecon.org"><span>Lectures in</span> Quantitative Economics</a></p>

			        	<p class="visuallyhidden"><a href="#skip">Skip to content</a></p>

			        	<ul class="site-authors">
			        		<li><a href="http://www.tomsargent.com/">Thomas J. Sargent</a></li>
			        		<li><a href="http://johnstachurski.net/">John Stachurski</a></li>
			        	</ul>

		        	</div>

		        	<nav class="main-nav">

			        	<ul>
		        			<li class="section-home"><a href="/">Home</a></li>
		        			<li class="section-py"><a href="/py/">Python</a></li>
		        			<li class="section-jl"><a href="/jl/">Julia</a></li>
							<li><a href="#pdf-options" name="pdf-options" rel="leanModal">PDF</a></li>
			        	</ul>

					</nav>

		        	<div class="pdf-options" id="pdf-options">
		        		<a class="modal_close" href="#"><span class="icon icon-cross"></span></a>
		        		<h2>Download PDF</h2>
		        		<ul>
		        			<li class="py"><a href="/_static/pdfs/QuantEconlectures-python3.pdf"><span class="thumb"><img src="/_static/img/py-logo.png" alt=" "></span> <span class="title">Python Lectures<span class="icon icon-chevron-right"></span></span></a></li>
		        			<li class="jl"><a href="/_static/pdfs/QuantEconlectures-julia.pdf"><span class="thumb"><img src="/_static/img/jl-logo.png" alt=" "></span> <span class="title">Julia Lectures<span class="icon icon-chevron-right"></span></span></a></li>

		        		</ul>
		        	</div>

		        	<div class="header-search"><gcse:searchbox-only></gcse:searchbox-only></div>

		        	<p id="combined_percentage" class="combined-coverage"><a href="/status.html"></a></p>

				</div>

	        </header>


			<div class="main">				<div class="breadcrumbs clearfix">
					<ul>
						<li><a href="https://quantecon.org/">Org</a> •</li>
						<li><a href="https://lectures.quantecon.org">Lectures</a> &raquo;</li>
				      
				      
				         <li><a href="index.html">Home</a> &raquo;</li>
				      
				      
				      
				         <li><a href="index_time_series_models.html">Time Series Models</a> &raquo;</li>
				      
				      
						<li>Covariance Stationary Processes</li>
					</ul>
				</div>


				<div class="content">

					<div id="skip"></div>

				    <div class="document">

{%- endblock header-%}

{% block codecell %}
<div class="cell border-box-sizing code_cell rendered">
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
<div class="cell border-box-sizing text_cell rendered">
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

				</div>				<div class="topic-pager clearfix">
					<ul>
						<li class="next"><a href="estspec.html" title="next chapter"><strong>Next topic</strong> <span>Estimation of Spectra</span></a></li>
						<li class="prev"><a href="index_time_series_models.html" title="previous chapter"><strong>Previous topic</strong> <span>Time Series Models</span></a></li>
					</ul>
				</div>

			</div>

			<footer class="footer">

				<div class="container">

					<p><a rel="license" href="http://creativecommons.org/licenses/by-nd/4.0/"><img alt="Creative Commons License" src="https://i.creativecommons.org/l/by-nd/4.0/80x15.png" /></a></p>

					<p>This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nd/4.0/">Creative Commons Attribution-NoDerivatives 4.0 International License</a>.</p>

					<p>&copy; Copyright 2017, Thomas J. Sargent and John Stachurski. Created using <a href="http://www.sphinx-doc.org/">Sphinx</a>, hosted with <a href="https://aws.amazon.com/">AWS</a>.</p>

				</div>

			</footer>

        </div>

        <div class="page-tools">
	        <ul>
		        <li class="top"><a href="#top" title="Back to top"><span class="icon icon-chevron-up"></span></a></li>
		        <li><a href="http://twitter.com/intent/tweet?url=https%3A%2F%2Flectures.quantecon.org%2Fpy/arma.html&via=QuantEcon&text=Covariance Stationary Processes" title="Share on Twitter" target="_blank"><span class="icon icon-twitter"></span></a></li>
		        <li><a href="https://www.linkedin.com/shareArticle?mini=true&url=https://lectures.quentecon.org%2Fpy/arma.html&title=Covariance Stationary Processes&summary=This%20website%20presents%20a%20series%20of%20lectures%20on%20quantitative%20economic%20modeling,%20designed%20and%20written%20by%20Thomas%20J.%20Sargent%20and%20John%20Stachurski.&source=QuantEcon" title="Share on LinkedIn" target="_blank"><span class="icon icon-linkedin"></span></a></li>
		        <li><a href="https://www.facebook.com/sharer/sharer.php?u=https%3A//lectures.quantecon.org%2Fpy/arma.html" title="Share on Facebook" target="_blank"><span class="icon icon-facebook"></span></a></li>
		        <li><span class="title">Share page</span></li>
	        </ul>
        </div>

        <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.0/jquery.min.js"></script>
        <script>window.jQuery || document.write('<script src="https://lectures.quantecon.org/_static/js/vendor/jquery-1.11.0.min.js"><\/script>')</script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

        <script src="https://lectures.quantecon.org/_static/js/plugins.js"></script>
        <script src="https://lectures.quantecon.org/_static/js/qe.js?v=1.5"></script>

	    <script>
	      var DOCUMENTATION_OPTIONS = {
	        URL_ROOT:    '',
	        VERSION:     '2018-Aug-8',
	        COLLAPSE_INDEX: false,
	        FILE_SUFFIX: '.html',
	        HAS_SOURCE:  true
	      };
	    </script>

		<script>
		  (function() {
		    var cx = '006559439261123061640:rgnmeebflcy';
		    var gcse = document.createElement('script');
		    gcse.type = 'text/javascript';
		    gcse.async = true;
		    gcse.src = 'https://cse.google.com/cse.js?cx=' + cx;
		    var s = document.getElementsByTagName('script')[0];
		    s.parentNode.insertBefore(gcse, s);
		  })();
		</script>

		<script>
		  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
		  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
		  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
		  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');
		  ga('create', 'UA-54984338-1', 'auto');
		  ga('send', 'pageview');
		</script>

    </body>
</html>


{%- endblock footer-%}