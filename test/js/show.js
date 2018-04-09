(function () {

	$("button#back").click(function(e){
	  window.location.href="grader.html";
	});

	var data = localStorage.getItem("data");
	if(data){
		data = JSON.parse(data);
	};

	$("#ori-title").text(data['ori']['title']);
	$("#ori-content").html(data['ori']['text']);
	var restitle = "";
	for(i=0;i<data['res']['title'].length;i++){
		var weight = 200-parseInt(205*data['res']['tit_att'][i]);
		restitle = restitle+"<span popover-top='"+data['res']['tit_att'][i]+"' style='color:rgb("+weight+","+weight+","+weight+");'>"+data['res']['title'][i]+"</span> "
	}
	var restext = "";
	for(i=0;i<data['res']['content'].length;i++){
		var weight = 200-parseInt(205*data['res']['con_att'][i]);
		restext = restext+"<span popover-top='"+data['res']['con_att'][i]+"' style='color:rgb("+weight+","+weight+","+weight+");'>"+data['res']['content'][i]+"</span> "
		if($.inArray(i, data['res']['conbr'])>=0)
			restext += "<br/>";
	}
	restext += "<br/>"
	for(i=0;i<data['res']['code'].length;i++){
		var weight = 200-parseInt(205*data['res']['cod_att'][i]);
		restext = restext+"<span popover-top='"+data['res']['cod_att'][i]+"' style='color:rgb("+weight+","+weight+","+weight+");'>"+data['res']['code'][i]+"</span> "
		if($.inArray(i, data['res']['codbr'])>=0)
			restext = restext+"<br/>";
	}
	$("#res-title").html(restitle);
	$("#res-content").html(restext);
	var tags = data['ori']['tag'].split(' ');
	for (i = tags.length-1; i >-1; i--) {
		if(tags[i])
	    $("#ori-content").after("<span class='badge oritag'>"+tags[i]+"</span>");
		$("#res-content").after("<span class='badge oritag'>"+tags[i]+"</span>");
	};

})();
