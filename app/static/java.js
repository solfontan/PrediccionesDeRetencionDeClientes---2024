document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');

    form.addEventListener('submit', function(event) {
        event.preventDefault(); // Evita que el formulario se envíe

        // Obtiene los valores de correo electrónico y contraseña
        const email = document.getElementById('exampleInputEmail1').value;
        const password = document.getElementById('exampleInputPassword1').value;
        const country = document.getElementById('countrySelect').value;

        // Lógica de autenticación de ejemplo (puedes reemplazarla con tu lógica real)
        if (email === 'example@gmail.com' && password === 'password' && country === 'france') {
            window.location.href = '/france_page'; // Redirige a la página principal si la autenticación es exitosa
        } else if (email === 'example@gmail.com' && password === 'password' && country === 'spain') {
            window.location.href = '/spain_page';
        } else if (email === 'example@gmail.com' && password === 'password' && country === 'germany') {
            window.location.href = '/germany_page';
        } else {
            alert('Contraseña incorrecta'); // Muestra un mensaje de advertencia si la autenticación falla
        }
    });
});
