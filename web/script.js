const URL = "https://mbllj30k-5500.brs.devtunnels.ms//tm-my-image-model/";

let model, webcam, labelContainer, maxPredictions;
let isVideoActive = false;

async function toggleVideo() {
    const btn = document.getElementById('detection-zone__camera-button');
    const container = document.getElementById("detection-zone__camera");
    labelContainer = document.getElementById("detection-zone__label-container");

    if(!isVideoActive){
        try{
            btn.innerText = "Cargando modelo...";

            if (!model) {
                const modelURL = URL + "model.json";
                const metadataURL = URL + "metadata.json";
                model = await tmImage.load(modelURL, metadataURL);
                maxPredictions = model.getTotalClasses();
            }
            
            webcam = new tmImage.Webcam(500, 500, false); 
            await webcam.setup({ facingMode: "environment" });
            await webcam.play();
            
            // Iniciamos el bucle de predicción
            window.requestAnimationFrame(loop);

            container.appendChild(webcam.canvas);

            labelContainer.innerHTML = "";
            for (let i = 0; i < maxPredictions; i++) {
                const contenedorClase = document.createElement("div");
                contenedorClase.innerHTML = `
                    <span></span> 
                    <progress value="0" max="100"></progress>
                `;
                labelContainer.appendChild(contenedorClase); 
            }

            btn.innerText = "Apagar Cámara";
            isVideoActive = true;
        } catch (err) {
            alert("Error detallado: " + err); // Esto te dirá si es permiso o falta de soporte
            console.error(err);
        }
    } else {
        if (webcam) {
            webcam.stop();
        }
        labelContainer.innerHTML = "";
        
        container.innerHTML = "";
        btn.innerText = "Encender Cámara";
        isVideoActive = false;
    }
}

/*async function init() {
    // Cambiamos el texto del botón para saber que está cargando
    const btn = document.getElementById('detection-zone__camera-button');
    btn.innerText = "Cargando modelo...";

    const modelURL = URL + "model.json";
    const metadataURL = URL + "metadata.json";

    try {
        model = await tmImage.load(modelURL, metadataURL);
        maxPredictions = model.getTotalClasses();

        const flip = true; 
        webcam = new tmImage.Webcam(200, 200, flip); 

        await webcam.setup(); 
        await webcam.play();
        window.requestAnimationFrame(loop);

        // Agregamos el canvas al div
        document.getElementById("detection-zone__camera").appendChild(webcam.canvas);
        
        labelContainer = document.getElementById("detection-zone__label-container");
        labelContainer.innerHTML = ""; // Limpiar antes de agregar
        for (let i = 0; i < maxPredictions; i++) {
            labelContainer.appendChild(document.createElement("div"));
        }

        btn.innerText = "Cámara Activa";
    } catch (error) {
        console.error("Error al cargar el modelo o cámara:", error);
        alert("Asegúrate de estar usando un servidor (http://localhost) y no abriendo el archivo directamente.");
    }
}*/

async function loop() {

    if (!isVideoActive) return; 

    webcam.update(); 
    await predict();
    window.requestAnimationFrame(loop);
}

async function predict() {

    if (!isVideoActive) return; 

    const prediction = await model.predict(webcam.canvas);
    for (let i = 0; i < maxPredictions; i++) {
        const porcentaje = (prediction[i].probability * 100).toFixed(2);
        const contenedor = labelContainer.childNodes[i];
        const barra = contenedor.querySelector('progress'); // Guardamos la barra en una variable

        contenedor.querySelector('span').innerText = `${prediction[i].className}: ${porcentaje}%`;
        barra.value = porcentaje;

        if (prediction[i].className === "Chagas") {
            barra.style.accentColor = "#eb7507";
        } else {
            barra.style.accentColor = "#07eb0b";
        }
    }
}  

const btn = document.getElementById('detection-zone__camera-button');
btn.addEventListener('click', toggleVideo);