import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import datetime
import re

# Set up Chrome options for headless browsing
chrome_options = Options()

chrome_options.add_argument('--ignore-ssl-errors=yes')
chrome_options.add_argument('--ignore-certificate-errors')
# chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--disable-gpu")  # Disable GPU for headless mode

# Set up WebDriver (you may need to download ChromeDriver or use WebDriverManager to manage the driver automatically)
driver = webdriver.Chrome(options=chrome_options)
print(driver.service.path)
# URL to scrape
url = 'https://localhost:8080/actualSystemData/actualSystemData.html'

# Open the page
driver.get(url)
driver.delete_all_cookies()
driver.add_cookie({'name': 'lang', 'value': 'fr'})
driver.add_cookie({'name': 'sessionid',
                   'value': 'SmvGTWSpS3IzAFqz0CyHmnGcdm79dHW3e9hNTXSZHWqsB4o3snSBVQkCTcxuB9gEh7o0flsWp7ITehbvQssYfdOjiVerlknzRZ3e'})
driver.add_cookie({'name': 'nav_offset', 'value': '0'})

driver.get(url)


# Wait for the page to load properly (adjust the time if needed)
print("waiting for the page to load")
time.sleep(10)
# print("page loaded: ", driver.page_source)
# Open or create a CSV file for writing
with open('scraped_data.csv', mode='w', newline='') as file:
    print("opened csv file")
    writer = csv.writer(file)
    # Write the header row in CSV
    csv_header = ["timestamp", "numéro du cordon", "numéro de pièce", "numéro de serie de la pièce", "intensité", "tension", "vitessedévidage", "Courant", "Tension", "Vitesse de dévidage", "Tension de soudage", "Vitesse de dévidage", "Courant de soudage MMA", "Dynamique MMA", "CEL intensité", "CEL dynamique", "Intensité I1 TIG", "Courant de descente TIG", "TIG balance", "Valeur commande courant", "Valeur commande dévidoir", "TIG courant d'amoçage", "TIG courant d'extinction", "TIG diamètre d'électrode", "TIG rampe amorçage", "TIG rampe d'extinction", "Correction de hauteur d’arc", "Courant hotstart MMA", "CEL Hostart intensité", "Correction dynamique/pulsé", "Dynamique", "puissance", "Stabilisateur de hauteur d’arc (PMC)", "Stabilisateur de pénétration", "énergie", "intensité moteur 3", "intensité moteur 2", "Intensité moteur 3", "M1 courant moteur", "M2 courant moteur", "M3 courant moteur", "Pression du crossjet", "débit eau", "MIG gaz rv", "MIG gaz rv", "Gaz total consommé", "température eau", "Heures activation débit", "Heures générateur ON"]
    writer.writerow(csv_header)

    # Function to scrape values and store them in CSV
    try:
        last_values = {}
        while True:
            # Scrape the current values
            cells = driver.find_elements(By.CLASS_NAME, 'tableElement')
            values_dict = {}
            for cell in cells:
                title = cell.find_element(By.TAG_NAME, "div").get_attribute("title")
                content = re.findall(r"[-+]?(?:\d*\.*\d+)", cell.get_attribute("textContent"))
                value = None if len(content) == 0 else float(content[0])
                if not title == '' and not content == '':
                    values_dict[title] = value

            if not values_dict == last_values:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([current_time]+list(values_dict.values()))
                print(values_dict)
                last_values = values_dict

            # Sleep for some time before checking again (adjust interval as needed)
            print('\n\nwaiting for next change')
            time.sleep(1)

    except KeyboardInterrupt:
        print("Scraping stopped by user.")

    except Exception as e:
        print("--------EXCEPTION---------\n", e)

    finally:
        # Clean up and close the driver
        driver.quit()
