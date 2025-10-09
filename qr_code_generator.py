# let's make a qr code generator using python
import qrcode
import os

data = input("Enter the text or url:").strip()
filename = input("Enter the filename (without extension):").strip()

# Ensure the file name has .png extension
if not filename.lower().endswith('.png'):
    filename += '.png'

qr = qrcode.QRCode(box_size=10, border=4)
qr.add_data(data)
qr.make(fit=True)  # ENsures the QR code is properly sized
image = qr.make_image(fill_color="black", back_color="white")
image.save(filename)
print(f'QR code saves as {os.path.abspath(filename)}')
