// static/script.js
document.addEventListener("DOMContentLoaded", function() {
    const blockContainer = document.getElementById('block-container');
    const addLayerButton = document.getElementById('add-layer');
    const trainButton = document.getElementById('train-network');
    
    let layerCounter = 0;
    layersInfo = {};

    addLayerButton.addEventListener('click', function() {
        const layerType = prompt("Enter layer type (e.g., Affine, ReLU, Softmax):");
        const inputSize = prompt("Enter input size:");
        const outputSize = prompt("Enter output size:");

        if (layerType && inputSize && outputSize) {
            let layerId = null;
            if (layerType == "Affine"){
                layerId = `${layerType}${++layerCounter}`;
            }
            else {
                layerId = `${layerType}${layerCounter}`;
            }
            layersInfo[layerId] = [parseInt(inputSize), parseInt(outputSize)];
            
            const layerBlock = document.createElement('div');
            layerBlock.className = 'layer-block';
            layerBlock.textContent = `${layerId}: (${inputSize}, ${outputSize})`;
            blockContainer.appendChild(layerBlock);
        }
    });

    trainButton.addEventListener('click', function() {
        fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ layers_info: layersInfo })
        })
        .then(response => response.json())
        .then(result => {
            console.log(result);
            alert("Training complete! Check console for details.");
        });
    });
});
