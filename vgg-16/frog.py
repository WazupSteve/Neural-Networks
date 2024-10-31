import requests

def download_cat_image(url, filename):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f"Image downloaded and saved as {filename}")
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage ( replace with your image url)
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSD0R7D0xWDwkKy3HJVbH_k-926TnLtSFa1mA&s"  
filename = "test.jpeg"
download_cat_image(url, filename)