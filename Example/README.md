PARP : Plataforma abierta restauración películas ![GPL2 License](http://img.shields.io/:license-gpl2-blue.svg?style=flat-square)
===========


Ejemplo
-------

Este ejemplo está dirigido a usuarios con cierta experiencia en programación en C++, que deseen programar nuevos plugins para Natron y OpenFX, utilizando funciones de la librería de computer vision OpenCV.

En particular en el código se incluyen las funciones de lectura y escritura entre ambos estándares.
Al video de entrada se aplica la función threshold de OpenCV para umbralizar imágenes, con un único paramétro de entrada para establecer dicho umbral.


### Compilación en Linux
Ejecutar los siguientes comandos:

-`git clone https://github.com/SebastianBugna/PARP`
-`cd Example`
-`make CONFIG=release`

Se crea subdirectorio llamado Linux-64-realease. El plugin se encuentra dentro de este subdirectorio en la carpeta `*.bundle`.
Para cargar el plugin, desde Natron ir a: `Edit>>Preferences...>>Plug-ins>>OpenFX-Plugins search path` y agregar la ruta a el directorio `*.bundle`

En Natron el plugin debe aparecer en el grupo de Nodos "ExampleParp", y dentro de dicho grupo se crea el plugin "Example"


