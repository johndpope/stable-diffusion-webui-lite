NOTE: 
  - 'ngrok.exe' and 'start.py' ae for local use
  - 'API_KEY.txt', 'flask_app.py' should be uplaoded to your remote pythonanywhere.com site

1. rename 'API_KEY.txt' to and paste in your ngrok APK_KEY (remember that API_KEY is not the AUTH_TOKEN, it should be manually created in your dashboard)
2. open browser and login in your pythonanywhere.com control panel 
2. in the File tab, upload 'API_KEY.txt', 'flask_app.py' to your default WWWROOT (tipically `/home/<username>/flask_app/`)
3. in the Webapps tab, reload your website and visit 'https://<username>.pythonanywhere.com/ngrok/refresh' to check it works
