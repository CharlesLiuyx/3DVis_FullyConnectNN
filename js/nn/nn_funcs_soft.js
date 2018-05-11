function getNNOutput() {
	var imageData = tinyCtx.getImageData(0, 0, 28, 28);
	
	var data = imageData.data;
	
	var pixel = 0;
	var input = new Array(nPixels);
	//console.log('here i am');
	//console.log(data.length);
	
		for(var i = 0, n = data.length; i < n; i += 4) {
			if (goodStart){
				if (data[i]) {
				  input[pixel] = ((data[i]/255)*1.275)-0.1;// * 1.175;
				}
				else {
				  input[pixel] = -0.1;
				}
			}
			allNodeInputs[pixel] = data[i];
			allNodeOutputs[pixel] = input[pixel];
			allNodeNums[pixel] = pixel+1;
			pixel++;
		}
	
	var input32 = reshapeArray(input);
	var inp = Vector.create(input32);
	
	var hidden_outputs_1 = Vector.Zero(nHiddenNodes_1);
	var hidden_outputs_1a = new Array(nHiddenNodes_1);
	var hidden_outputs_2 = Vector.Zero(nHiddenNodes_2);
	var hidden_outputs_2a = new Array(nHiddenNodes_2);
	var final_outputsa = new Array(nFinalNodes);
	
	for (i=1; i<=nHiddenNodes_1; i++){
	  var weights = hidden_weights_1.row(i);
	  var sum = inp.dot(weights);
	  sum += hidden_biases_1.e(i);
	  hidden_outputs_1a[i-1] = sigma(sum);
	  allNodeInputs[nPixels+i-1] = sum;
	  allNodeOutputs[nPixels+i-1]=hidden_outputs_1a[i-1];
	  allNodeNums[nPixels+i-1] = i;
	  
	}
	hidden_outputs_1.setElements(hidden_outputs_1a);
	
	for (i=1; i<=nHiddenNodes_2; i++){
		  var weights = hidden_weights_2.row(i);				  
		  var sum = hidden_outputs_1.dot(weights);
		  sum += hidden_biases_2.e(i);
		  hidden_outputs_2a[i-1] = sigma(sum);
		  allNodeInputs[nPixels+nHiddenNodes_1+i-1]=sum;
		  allNodeOutputs[nPixels+nHiddenNodes_1+i-1]=hidden_outputs_2a[i-1];
		  allNodeNums[nPixels+nHiddenNodes_1+i-1] = i;
	}
	hidden_outputs_2.setElements(hidden_outputs_2a);
	
	var sums = final_weights.x(hidden_outputs_2);
	var newSums = sums.add(final_biases);
	
	for (i=1; i<=nFinalNodes; i++){
		final_outputsa[i-1] = sigma(newSums.e(i));
		allNodeInputs[nPixels+nHiddenNodes_1+nHiddenNodes_2+i-1]=newSums.e(i);
		allNodeOutputs[nPixels+nHiddenNodes_1+nHiddenNodes_2+i-1]=final_outputsa[i-1];
		allNodeNums[nPixels+nHiddenNodes_1+nHiddenNodes_2+i-1] = i;
	}
	
	allNodeOutputsRaw = allNodeOutputs.slice();
	normalizeWithinLayer(allNodeOutputs);
	
	if (!allZeroes){ 
		var ind1 = maxInd(final_outputsa);
		finalOutputID = nPixels+nHiddenNodes_1+nHiddenNodes_2+i-1 + ind1 - 10;
		final_outputsa[ind] = -10;
		var ind2 = maxInd(final_outputsa);
		document.getElementById("ans1").innerHTML = ind1;
		document.getElementById("ans2").innerHTML = ind2;
	} else {
		document.getElementById("ans1").innerHTML = "";
		document.getElementById("ans2").innerHTML = "";
	}
	
	isComputed = true;
	
	updateCubes();
	updateEdges();
	
	
	//console.log(imageData);
	imageData.data = null;
	imageData = null;
	//console.log(imageData);
};

function sigma(x) {
	//return 1.7159*math.tanh(0.6667*x);
	return 1/(1+math.exp(-x));
}
function reshapeArray(arr){
	// The input array walks along pixels ltr ltr ltr.
	// For proper input, we need it to walk ttb ttb ttb.
	var arr2 = new Array(1024);
	for (count = 0; count < 1024; count++){
		arr2[count] = -0.1;
	}
	for (count = 0; count < 768; count++){
		var row = math.floor(count/28)+2;
		var col = (count)%28+2;
		var newInd = col*32 + row;
		arr2[newInd] = arr[count];
	}
	return arr2;
}
function maxInd(arr) {
	ind = 0;
	val = arr[0];
	for (i=1; i<arr.length; i++){
		if (arr[i]>val){
			ind = i;
			val = arr[i];
		}				
	}
	return ind;
}
function normalizeWithinLayer(arr) {
	var len = arr.length;

	var minPixel = 100;
	var minHidden1 = 100;
	var minHidden2 = 100;
	var minFinal = 100;
	
	var maxPixel = -100;
	var maxHidden1 = -100;
	var maxHidden2 = -100;
	var maxFinal = -100;
	for (var i=0;i<len;i++){
		if (i<nPixels) {
			if (arr[i]>maxPixel)
				maxPixel = arr[i];
			else if (arr[i]<minPixel)
				minPixel = arr[i];
		} else if (i < nPixels+nHiddenNodes_1) {
			if (arr[i]>maxHidden1)
				maxHidden1 = arr[i];
			else if (arr[i]<minHidden1)
				minHidden1 = arr[i];
		} else if (i < nPixels+nHiddenNodes_1+nHiddenNodes_2) {
			if (arr[i]>maxHidden2)
				maxHidden2 = arr[i];
			else if (arr[i]<minHidden2)
				minHidden2 = arr[i];
		} else {
			if (arr[i]>maxFinal)
				maxFinal = arr[i];
			else if (arr[i]<minFinal)
				minFinal = arr[i];
		}
	}
	if (minPixel==maxPixel){					
		allZeroes = true;
		for (var i=0;i<len;i++){
			arr[i] = 0;					
		}
	} else {
		allZeroes = false;
		for (var i=0;i<len;i++){
			if (i<nPixels) {
				arr[i] = (arr[i] - minPixel)/(maxPixel-minPixel);
			} else if (i < nPixels+nHiddenNodes_1) {
				arr[i] = (arr[i] - minHidden1)/(maxHidden1-minHidden1);
			} else if (i < nPixels+nHiddenNodes_1+nHiddenNodes_2) {
				arr[i] = (arr[i] - minHidden2)/(maxHidden2-minHidden2);
			} else {
				arr[i] = (arr[i] - minFinal)/(maxFinal-minFinal);
			}
		}
	}
}

function loadData() {
	var nodeCount = 0;
	$.getJSON('./js/nn/webgl_300_100_flat_small_t.json',function(data){
		$.each(data.nodes,function(i,node){
			posX[nodeCount] = node.x;
			posY[nodeCount] = node.y;
			posZ[nodeCount] = node.z;
			nodeCount++;
		});
	}).error(function(){
		console.log('error');
	}).done(function(){
		createText();
		drawCubes();
		drawNegEdges();
		drawEdges();		
	});
}
