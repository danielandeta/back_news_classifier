import base64

with open("./images/logoTaws.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
print("gnsj")
print("\n")
print(encoded_string)
print("\n")



