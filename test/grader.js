(function () {

	chrome.browserAction.onClicked.addListener(function (tab) {
		chrome.tabs.create({url: "grader.html"});
	});

	$("button#form").click(function(e){
	  window.location.href="form.html";
	});

	$("button#back").click(function(e){
	  window.location.href="grader.html";
	});

	$("button#goonline").click(function(e){
	  var idsearch = $("#idsearch").val()
	  if(idsearch){
		  $.ajax ({
		  	     async:false,
	             url:"http://127.0.0.1:8000/aes/",
	             type:'POST',
	             dataType:'json',
	             data:{id:idsearch},
	             timeout:20000,
	             error:function(e){
	               alert("error")
	             },success:function(ljdata){
					 localStorage.setItem("data", JSON.stringify(ljdata));
		   		     window.location.href="show.html";
		         }
	      });
	  }else{
	  	alert('blank');
	  }

	});

 	var t = document.getElementsByClassName("text");
 	var i;
	for (i = 0; i < t.length; i++) {
	    autosize(t[i]);
	};

	$("#clear").click(function(e){
		$("input").val("");
		$("textarea").val("");
		$(".alert").remove();
		$(".tagmargin").remove();
		$("#tagin").show();
	});

	$("#tagin").keyup(function(event){
	    if(event.which==13){
	      var text = $("#tagin").val().replace(/(^\s*)|(\s*$)/g, "");
	      var tagnum = $(".alert").length;
	      if(text){
	      	if(tagnum==4){
	      		$("#tagin").hide();
	      	};
	      	$("#tagin").before("<div class='alert alert-primary sm-8 col' id='tag"+tagnum+"'>"+text+"</div>\
	      		<div class='sm-3 col tagmargin' id='margin"+tagnum+"'></div>");
	      	$("#tagin").val("");
	      	$("#tag"+tagnum).click(function(e){
	      		$("#tag"+tagnum).remove();
	      		$("#margin"+tagnum).remove();
	      		$("#tagin").show();
	      	});
	      };
	 	};
	});

})();
