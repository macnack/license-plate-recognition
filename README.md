# license-plate-recognition
Projekt zaliczeniowy przedmiot SW. Autor Maciej Krupka

## Funkcje do wykrywania tablicy:

1. `preprocess_image(image)`: Przygotowuje obraz do dalszej analizy poprzez zmniejszenie rozmiaru, konwersję do odcieni szarości, zastosowanie bilateralnego filtra, adaptacyjne progowanie, operacje morfologiczne. Zwraca obraz przetworzony oraz oryginalny obraz.
2. `chech_ratio(ratio)`:  Sprawdza, czy stosunek szerokości do wysokości tablicy jest w akceptowalnym zakresie. Zwraca wartość logiczną.
3. `check_license_plate_width(width, image_width)`: Sprawdza, czy szerokość tablicy przekracza jedną trzecią szerokości obrazu. Zwraca wartość logiczną.
4. `check_size_distortion(diff_height, diff_width)` : Sprawdza, czy wymiary bboxa nie sa zbyt zniekształcone. Zwraca wartość logiczną.
5. `detect_license_plate(image)`: Wykrywa tablicę rejestracyjną na obrazie. Wykorzystuje funkcje `preprocess_image` do przetworzenia obrazu, znajduje kontury, filtruje kontury tablic rejestracyjnych na podstawie określonych kryteriów (stosunek szerokości do wysokości, szerokość tablicy, zniekształcenie). Zwraca kontur tablicy rejestracyjnej oraz obraz przetworzony.

## Funkcje do rozpoznawania znaków:

1. `convert_letters(string)`: Konwertuje niektóre litery na cyfry w ciągu znaków tablicy rejestracyjnej. Zwraca przekonwertowany ciąg znaków.
2. `create_license_plate_image(image,pts)`: Przekształca kontur tablicy rejestracyjnej na prostokąt i zwraca przekształcony obraz tablicy.
3. `sign_recognitions(image, points, char_images)`: Rozpoznaje znaki na tablicy rejestracyjnej. Wykorzystuje przekształcony obraz tablicy, stosuje przetwarzanie obrazu (konwersja do przestrzeni barw HSV, progowanie, zamknięcie morfologiczne), znajduje kontury znaków, dopasowuje wzorce znaków i przeprowadza rozpoznawanie. Zwraca rozpoznane znaki.
