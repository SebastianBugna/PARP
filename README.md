PARP : Plataforma abierta restauración películas ![GPL2 License](http://img.shields.io/:license-gpl2-blue.svg?style=flat-square)
===========

Proyecto de grado desarrollado en la Facultad de Ingeniería de la Universidad de la República de Uruguay, en el Instituto de Ingeniería Eléctrica ("Profesor Agustín Cisa") y en el Instituto de Computación (INCO). 
Fue desarrollado por los estudiantes de grado Sebastián Bugna y Juan Andrés Friss de Kereki.

Estos plugins OFX fueron implementados utilizando:
[Natron 2.2.7](http://natron.inria.fr), 
[OpenFX](http://openeffects.org), 
[OpenCV](http://opencv.org).

Detección semi-automática de cortes (ShotCutDetection)
-------
Permite detectar y segmentar correctamente una secuencia en sus distintos planos.
Utiliza el método de suma de diferencias absolutas (SDA) que dadas dos imágenes, calcula su distancia como la suma total de la diferencia píxel a píxel.
Opcionalmente se puede importar una lista de cortes en formato EDL para dividir en planos la secuencia.


Eliminación de scratches (RemoveScratches)
-------

Permite detectar y posteriormente reparar los scratches presentes en películas deterioradas.
La implementación del algoritmo de detección de scratches utiliza la propuesta del artículo: `Robust Automatic Line Scratch Detection in Films` de Andrés Almansa y colegas, que utiliza una metodología a contrario para detectar los scratches.
La restauración de los scratches se realiza mediante algoritmos de image inpainting disponibles en OpenCV.


Deflicker
-------
Permite corregir el efecto de flicker de luminancia en secuencias de video.
Se basa en los artículos `Implementation of the Midway Image Equalization` y `Stabilization of flicker-like effects in image sequences
through local contrast correction` de Julie Delon y colegas.

Ejemplo (Example)
-------
Se provee un ejemplo sencillo destinado a programadores que deseen implementar nuevos algoritmos utilizando OFX y OpenCV.


### Compilación en Linux
Se utilizan Makefiles, ejecutando en un terminal los comandos:
- `make [opciones]`
Las opciones más comunes son:
- `CONFIG=release` : permite compilar la verión final y optimizada.
- `CONFIG=debug` : permite compilar una versión sin optimizaciones, para hacer debugging.
- `CONFIG=relwithdebinfo` : permite compilar una versión optmizada y hacer debugging.


Ver el archivo `Makefile.master` en el directorio raíz para conocer más opciones de compilación.
Es posible compilarlos todos juntos desde el diretorio raíz, o compilar cada plugin por separado ingresando a su subdirectorio y ejecturando los mismos comandos. 

Los plugins compilados se ubican en subdirectorios llamados, por ejemplos: Linux-64-realease.
En cada uno de estos subdiretorios se crea un directorio `*.bundle`. 
Se puede cargar en Natron los plugins moviendo el directorio `*.bundle` a la carpeta `/usr/OFX/Plugins`.
Alternativamente se puede utilizar la opción en Natron: `Edit>>Preferences...>>Plug-ins>>OpenFX-Plugins search path` donde se agrega la ruta a el directorio `*.bundle`


Manual de Usuario
------
Se proveen instrucciones para utilizar e instalar todas las herramientas desarrolladas, así como diferentes funcionalidades de Natron que permiten realizar la restauración de secuencias.