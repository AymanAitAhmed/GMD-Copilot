from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

chrome_options = Options()
chrome_options.add_argument('--ignore-ssl-errors=yes')
chrome_options.add_argument('--ignore-certificate-errors')
chrome_options.add_argument("window-size=1200x600")
driver = webdriver.Chrome(chrome_options)

driver.get("https://localhost:8081/userManagement/login.html")
time.sleep(5)
print(driver.page_source)
username_field = driver.find_element(By.ID,"username")
username_field.clear()
username_field.send_keys("admin")

password_field = driver.find_element(By.ID,"password")
password_field.clear()
password_field.send_keys("admin")

login_button = driver.find_element(By.ID, "loginbutton")
login_button.click()

time.sleep(5)

ok_btn = driver.find_element(By.CLASS_NAME,"closesubwindow")
print(driver.page_source)
print(ok_btn.is_displayed())
driver.execute_script("arguments[0].click();",ok_btn)
time.sleep(10)





