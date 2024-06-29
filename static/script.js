// static/script.js
document.addEventListener("DOMContentLoaded", function() {
    const blockContainer = document.getElementById('block-container');
    const layerButtons = document.querySelectorAll('.layer-button');
    const trainButton = document.getElementById('train-network');
    const refreshButton = document.getElementById('refresh-network');
    const undoButton = document.getElementById('undo-layer');

    let layerCounter = 0;
    let layersInfo = {};
    const layerBlocks = [];

    layerButtons.forEach(button => {
        button.addEventListener('click', function() {
            const layerType = this.getAttribute('data-layer-type');
            addLayerBlock(layerType);
        });
    });

    function addLayerBlock(layerType) {
        let layerId = null;
        if (layerType === "Affine") {
            layerId = `${layerType}${++layerCounter}`;
        } else {
            layerId = `${layerType}${layerCounter}`;
        }
        
        const layerBlock = document.createElement('div');
        layerBlock.className = 'layer-block';
        layerBlock.innerHTML = `
            <div>${layerId}</div>
            <label for="input-size-${layerId}">Input Size:</label>
            <input type="number" id="input-size-${layerId}" class="layer-input">
            <label for="output-size-${layerId}">Output Size:</label>
            <input type="number" id="output-size-${layerId}" class="layer-output">
        `;
        blockContainer.appendChild(layerBlock);
        layerBlocks.push(layerBlock);
    }

    trainButton.addEventListener('click', function() {
        layersInfo = {};
        layerBlocks.forEach((block, index) => {
            const layerId = block.querySelector('div').textContent;
            const inputSize = parseInt(block.querySelector('.layer-input').value);
            const outputSize = parseInt(block.querySelector('.layer-output').value);
            if (!isNaN(inputSize) && !isNaN(outputSize)) {
                layersInfo[layerId] = [inputSize, outputSize];
            }
            else{
                alert("Null value exsists");
            }
        });

        const epochs = parseInt(document.getElementById('epochs').value);
        const batchSize = parseInt(document.getElementById('batch-size').value);
        const learningRate = parseFloat(document.getElementById('learning-rate').value);
        const momentum = parseFloat(document.getElementById('momentum').value);
        const optimizer = document.getElementById('optimizer').value;

        const trainingConfig = {
            epochs: epochs,
            batch_size: batchSize,
            learning_rate: learningRate,
            momentum: momentum,
            optimizer: optimizer
        };
        
        fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ layers_info: layersInfo, config: trainingConfig })
        })
        .then(response => response.json())
        .then(result => {
            console.log(result);
            alert("Training complete! Check console for details.");
        });
    });

    refreshButton.addEventListener('click', function() {
        blockContainer.innerHTML = '';
        layersInfo = {};
        layerCounter = 0;
        layerBlocks.length = 0;
    });

    undoButton.addEventListener('click', function() {
        if (layerBlocks.length > 0) {
            const lastBlock = layerBlocks.pop();
            blockContainer.removeChild(lastBlock);
            layerId = lastBlock.querySelector('div').textContent;
            if (layerId.includes("Affine"))
                layerCounter--;
            delete layersInfo[layerId];
        }
    });
});
