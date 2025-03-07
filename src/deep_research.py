import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from dotenv import load_dotenv

load_dotenv()

def deep_research(prompt: str) -> str:
    options = Options()
    options.binary_location = os.getenv("CHROME_CANARY_BINARY_LOCATION")

    options.add_argument(
        os.getenv("CHROME_CANARY_USER_DATA_PATH")
    )

    options.add_argument("--profile-directory=Default")
    # CanaryでのCookieやセッションが流用されているため、2段階認証などログインステップを省略できる


    driver = webdriver.Chrome(options=options)

    driver.get("https://gemini.google.com/app")

    open_modal_button = driver.find_element(By.XPATH, "/html/body/chat-app/main/div/bard-mode-switcher/button")
    open_modal_button.click()

    WebDriverWait(driver, 5).until(
        EC.visibility_of_element_located((By.CLASS_NAME, "mat-mdc-menu-content"))
    )
    deepresearch_mode_button = driver.find_element(By.XPATH, "/html/body/div[7]/div[2]/div/div/div/button[5]")
    deepresearch_mode_button.click()

    time.sleep(1)

    prompt_textarea = driver.find_element(By.XPATH, "/html/body/chat-app/main/side-navigation-v2/bard-sidenav-container/bard-sidenav-content/div[2]/div/div[2]/chat-window/div/input-container/div/input-area-v2/div/div/div[2]/div/div/rich-textarea")

    prompt_textarea.send_keys(prompt)

    prompt_textarea.send_keys(Keys.ENTER)

    WebDriverWait(driver, 60*5).until(
        EC.visibility_of_element_located(By.XPATH, "/html/body/chat-app/main/side-navigation-v2/bard-sidenav-container/bard-sidenav-content/div[2]/div/div[2]/chat-window/div/chat-window-content/div[1]/infinite-scroller/div/model-response/div/response-container/div/div[2]/div[2]/div/message-content/div/p/div/response-element/deep-research-confirmation-widget/div/div[4]/button[2]")
    )

    start_button = driver.find_element(By.XPATH, "/html/body/chat-app/main/side-navigation-v2/bard-sidenav-container/bard-sidenav-content/div[2]/div/div[2]/chat-window/div/chat-window-content/div[1]/infinite-scroller/div/model-response/div/response-container/div/div[2]/div[2]/div/message-content/div/p/div/response-element/deep-research-confirmation-widget/div/div[4]/button[2]")
    start_button.click()

    WebDriverWait(driver, 60*20).until(
        EC.visibility_of_element_located(By.XPATH, "/html/body/chat-app/main/side-navigation-v2/bard-sidenav-container/bard-sidenav-content/div[2]/div/div[2]/immersive-window/div/immersive-panel/extended-response-panel/toolbar/div/div[2]/button[1]/span[2]")
    )

    response_element = driver.find_element(By.XPATH, "/html/body/chat-app/main/side-navigation-v2/bard-sidenav-container/bard-sidenav-content/div[2]/div/div[2]/immersive-window/div/immersive-panel/extended-response-panel/div/response-container")

    response = response_element.get_attribute("innerHTML")

    return response

