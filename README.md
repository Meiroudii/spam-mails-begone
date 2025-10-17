# SpamMails_begone

How to activate the server
```
python app.py
```

Test runs
```

curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text":"Congratulations! You have won $1000. Click here to claim."}'


curl -X POST -F 'text=Free entry! Call now' http://127.0.0.1:5000/predict
```
