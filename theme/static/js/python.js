// python.js v1.0


// Declare MathJax Macros for the Appropriate Macros
MathJax.Hub.Config({
  TeX: {
    Macros: {
      Var:     "\\mathop{\\mathrm{Var}}",
      trace:   "\\mathop{\\mathrm{trace}}",
      argmax:  "\\mathop{\\mathrm{arg\\,max}}",
      argmin:  "\\mathop{\\mathrm{arg\\,min}}",
      proj:  "\\mathop{\\mathrm{proj}}",
      col:  "\\mathop{\\mathrm{col}}",
      Span:  "\\mathop{\\mathrm{span}}",
      epsilon: "\\varepsilon",
      EE: "\\mathbb{E}",
      PP: "\\mathbb{P}",
      RR: "\\mathbb{R}",
      NN: "\\mathbb{N}",
      ZZ: "\\mathbb{Z}",
      aA: "\\mathcal{A}",
      bB: "\\mathcal{B}",
      cC: "\\mathcal{C}",
      dD: "\\mathcal{D}",
      eE: "\\mathcal{E}",
      fF: "\\mathcal{F}",
      gG: "\\mathcal{G}",
      hH: "\\mathcal{H}",
      
    }
  }
});
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [ ['$','$'], ['\\(','\\)'] ],
    processEscapes: true
  }
});


// Status Page and Badge for Lecture Executability Testing

var status_data;
var last_test_time;

const LECTURE_OK = 0;
const LECTURE_FAILED = 1;
const LECTURE_ERROR = -1;

/**
    A function that interrogates the JSON data produced by the execution testing system
    to determine whether this lecture passed or failed the last test.

    The Javascript infers the name of the lecture file from the filename of the HTML file itself.
    If the HTML filename is not a lecture, or otherwise cannot be found in the JSON data,
    this function will return an error value.

    The return values are the constant expressions LECTURE_OK, LECTURE_FAILED and LECTURE_ERROR.
**/
function determine_page_status()
{
    var path = window.location.pathname;
    var filename_parts = path.split("/");
    var filename = filename_parts.pop();

    var lecture_name = filename.split(".")[0].toLowerCase();

    var res = LECTURE_ERROR;

    for (var i = 0; i < status_data.length; i ++)
    {
        if (status_data[i]['name'].split('/').pop() === lecture_name)
        {
            if (status_data[i]['result'] === 0)
            {
                res = LECTURE_OK;
            }
            else
            {
                res = LECTURE_FAILED;
            }
        }
    }

    return res;
}

function emptyNode(myNode)
{
    while (myNode.firstChild)
        myNode.removeChild(myNode.firstChild);
}

function regenerateTableHeaderRow(table)
{
    rawHTML = "<tr><th class='resultsTableHeader'>Lecture File</th><th class='resultsTableHeader'>Language</th><th>Running Time</th><th></th></tr>";
    table.innerHTML = rawHTML;
}

function insertNewRow(newRow)
{
    var table = document.getElementById("status_table");
    var row = table.insertRow(-1);
    row.setAttribute("id", "statusTableRow" + newRow['id'], 0);

    // Insert new cells (<td> elements) at the 1st and 2nd position of the "new" <tr> element:
    var lectureCell = row.insertCell(0);
    var langCell = row.insertCell(1);
    var runtimeCell = row.insertCell(2);
    var statusCell = row.insertCell(3);
    var badge, status, color, lang, link;

    if (newRow['result'] === 0)
    {
        status = "passing";
        color = "brightgreen";
    }
    else if (newRow['result'] === 1)
    {
        status = "failing";
        color = "red";
    }
    else if (newRow['result'] === -1) {
        status = "not available";
        color = "lightgrey";
    }

    link = '/' + newRow['extension'] + '/' + newRow['name'] + '.html';

    badge = '<a href="' + link + '"><img src="/_static/img/execution-test-' + status + '-' + color + '.svg"></a>';

    // Add some text to the new cells:
    lectureCell.innerHTML = newRow['name'];
    langCell.innerHTML = newRow['language'];
    runtimeCell.innerHTML = newRow['runtime'];
    statusCell.innerHTML = badge;
}

function updateTable()
{
    // empty the table
    table = document.getElementById("status_table");
    emptyNode(table);
    regenerateTableHeaderRow(table);

    // add the data
    for (var i = 0; i < status_data.length; i ++)
    {
        insertNewRow(status_data[i]);
    }
}


function load_status_table()
{
    $.getJSON( "_static/code-execution-results.json", function( data )
    {
        status_data = [];
        last_test_time = data.run_time;
        $('#last_test_time').text(last_test_time);
        for (var key in data.results)
        {
            var new_record = {};
            new_record['name'] = data.results[key].filename;
            new_record['runtime'] = data.results[key].runtime;
            new_record['extension'] = data.results[key].extension;
            new_record['result'] = data.results[key].num_errors;
            new_record['language'] = data.results[key].language;

            status_data.push(new_record);
        }

        updateTable();
    })
    .error( function() {
      console.log('Error reading code execution results JSON')
      updateTable();
    });
};


/**
    Updates the "badge" on the lecture page with the result of the last execution test.
    page_status will contain one of three constant values:

        LECTURE_OK:     The code passed the last test successfully.
        LECTURE_FAILED: The lecture did not pass the test.
        LECTURE_ERROR:  Either this page is not a lecture that's been tested, or something's gone
                        wrong sonewhere; either way, do not display a badge.
**/
function update_page_badge(page_status)
{
    var badge = document.getElementById("executability_status_badge");
    var status, color;

    if (page_status === LECTURE_OK)
    {
        status = "passing";
        color = "brightgreen";
    }
    else if (page_status == LECTURE_FAILED)
    {
        status = "failing";
        color = "red";
    }
    else if (page_status == LECTURE_ERROR)
    {
        status = "not available";
        color = "lightgrey";
    }
    else
    {
        console.log("Panic! Invalid parameter passed to update_page_badge().");
    }

    badge.setAttribute("src", "/_static/img/execution-test-" + status + "-" + color + ".svg");

    badge.style.display="block";

    return;
}


/**
    Updates the badge on the front page of the lectures. This badge displays the number of lectures
    which passed the last execution test relative to the number of lectures which were tested overall.

    percentage is an integer (or what passes for one in Javascript) which is guaranteed to be *either* between
    zero and one hundred inclusive, which is the value to display to the user, or an error code of -1 which indicates
    that either the JSON file did not load or that the data it contains resulted in a percentage which lies outside
    of those boundaries.
**/
function get_badge(percentage)
{
    var color, badge;

    if (percentage > -1)
    {
      if ( percentage < 50 ) {
        color = 'red';
      } else {
        color = 'brightgreen';
      }
      badge = '<img src="https://img.shields.io/badge/Total%20coverage-' + percentage + '%25-' + color + '.svg">';
    } else {
      badge = '<img src="https://img.shields.io/badge/Total%20coverage-not%20available-lightgrey.svg">';
    }
    return badge;
}


function load_this_page_badge()
{
  $.getJSON( "_static/code-execution-results.json", function( data )
  {
      status_data = [];
      for (var key in data.results)
      {
          var new_record = {};
          new_record['name'] = data.results[key].filename;
          new_record['runtime'] = data.results[key].runtime;
          new_record['extension'] = data.results[key].extension;
          new_record['result'] = data.results[key].num_errors;
          new_record['language'] = data.results[key].language;

          status_data.push(new_record);
      }

      var page_status = determine_page_status();
      update_page_badge(page_status);
  })
  .error( function( data ) {
    console.log('Error reading code execution results JSON')
    update_page_badge(-1);
  });
}



function load_percentages(lang)
{
  var number_of_lectures = {};
  var number_which_passed = {};
  var keys_list = [];
  var combined_percentage, py_percentage, jl_percentage;

  $.getJSON( "_static/code-execution-results.json", function( data )
  {
    for (var key in data.results)
    {
      if (data.results[key].num_errors === 0)
      {
        if (!(data.results[key].extension in number_which_passed))
        {
            number_which_passed[data.results[key].extension] = 0;
            keys_list.push(data.results[key].extension);
        }
        number_which_passed[data.results[key].extension] += 1;
      }

      if (!(data.results[key].extension in number_of_lectures))
      {
        number_of_lectures[data.results[key].extension] = 0;
      }
      number_of_lectures[data.results[key].extension] += 1;
    }

    var percentages = {};
    var total_lectures = 0;
    var total_passing = 0;
    for (var k in keys_list)
    {
        key = keys_list[k];

        percentages[key] = 0;
        if (number_of_lectures[key] === 0)
        {
            // An appropriate value for this is yet to be determined.
            percentages[key] = 100;
        }
        else
        {
            percentages[key] = Math.floor(100 * number_which_passed[key] / number_of_lectures[key]);
        }

        // Sensible boundary checking.
        if (percentages[key] < 0 || percentages[key] > 100)
        {
            percentages[key] = -1;
        }

        total_lectures += number_of_lectures[key];
        total_passing += number_which_passed[key];
    }

    if (total_lectures === 0)
    {
        combined_percentage = 0;
    }
    else
    {
        combined_percentage = Math.floor(100 * total_passing / total_lectures);
    }

    $('#combined_percentage a').html(get_badge(combined_percentage, 'Total'));
    $('#py_percentage a').html(get_badge(py_percentage, 'Python'));
    $('#jl_percentage a').html(get_badge(jl_percentage, 'Julia'));
    
  })
  .error( function( data ) {
    console.log('Error reading code execution results JSON');
  });
}

$(function () {

	/* Collapsed code block */
	$("div[class^='collapse']").each(function(){
		$('.highlight', this).after('<a href="#" class="toggle toggle-less" style="display:none;"><span class="icon icon-angle-double-up"></span><em>Show less...</em></a>');
		$('.highlight', this).after('<a href="#" class="toggle toggle-more"><span class="icon icon-angle-double-down"></span><em>Show more...</em></a>');
	});

	$('div[class^="collapse"]').on('click', '.toggle', function(e){
		var codeBlock = $(this).parents('div[class^="collapse"]');
    	if ( codeBlock.hasClass('expanded') ) {
    		codeBlock.removeClass('expanded').find('.toggle').toggle();
    		$('html, body').animate({
        		scrollTop: $(codeBlock).offset().top - 50
    		}, 400);
    	} else {
    		codeBlock.addClass('expanded').find('.toggle').toggle();
    	}
    	e.preventDefault();
	});

	/* Display a notebook execution badge on lecture pages */
	if ( $('#executability_status_badge').length > 0 ) {
		load_this_page_badge();
	}

	/* Display a notebook execution status table */
	if ( $('#status_table').length > 0 ) {
		load_status_table();
	}

	/* Display coverage badges */
	load_percentages();
    

});
