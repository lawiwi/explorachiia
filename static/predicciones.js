// Obtener los botones y las categorías
const todosBtn = document.getElementById("todosBtn");
const comidaBtn = document.getElementById("comidaBtn");
const ocioBtn = document.getElementById("ocioBtn");
const deportesBtn = document.getElementById("deportesBtn");

const comidaCategory = document.getElementById("comida");
const ocioCategory = document.getElementById("ocio");
const deportesCategory = document.getElementById("deportes");

// Función para mostrar las categorías
function showCategory(category) {
    comidaCategory.style.display = 'none';
    ocioCategory.style.display = 'none';
    deportesCategory.style.display = 'none';
    category.style.display = 'grid';
}

// Mostrar todas las categorías al presionar "Todos"
todosBtn.addEventListener("click", function() {
    comidaCategory.style.display = 'grid';
    ocioCategory.style.display = 'grid';
    deportesCategory.style.display = 'grid';
});

// Mostrar categoría de comida
comidaBtn.addEventListener("click", function() {
    showCategory(comidaCategory);
});

// Mostrar categoría de ocio
ocioBtn.addEventListener("click", function() {
    showCategory(ocioCategory);
});

// Mostrar categoría de deportes
deportesBtn.addEventListener("click", function() {
    showCategory(deportesCategory);
});


async function cargarDia(dia) {
    try {
        const response = await fetch(`/prediccion/${empresa}/dia/${dia}`);
        const data = await response.json();

        const detalleDiv = document.getElementById('detalle-visitas');
        detalleDiv.innerHTML = `
            <h2>Posibles visitas para ${dia}:</h2>
            <h1>${data.promedio.toFixed(2)}</h1>
            <img src="${data.grafico_url}" alt="Gráfico ${dia}" style="max-width: 600px;">
        `;
    } catch (error) {
        console.error("Error cargando el día:", error);
    }
}
