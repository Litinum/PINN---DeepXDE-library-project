# PINN---DeepXDE-library-project
Za diplomski

Dependency:
	- deepxde
	- numpy
	- matplotlib
	- threading
	- os
	- tkinter
	- customtkinter
	- PIL
	
Pokretanje:
	1. Pokrenuti "oscilator.py" skriptu
	2. Nakon sto se otvori GUI, na levoj strani je neophodno uneti fizicke parametre ili odabrati neki od presets (TBA)
		a) U gornjem levom uglu se nalazi dugme za dodatna podesavanja mreze
	3. Nakon zavrsenih podesavanja parametara, kliknuti na dugme "Start"
	4. Nakon sto se treniranje zavrsi, u centralnom prozoru ce se pokrenuti simulacija dok sa desne strane ce se prikazati graf kretanja oscilatora tokom vremena
	5. Pritiskom na dugme "Play" u centralnom prozoru, simulacija se ponavlja (bez ponovnog treniranja)
