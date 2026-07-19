#!/usr/bin/env python3
import urllib.request
import urllib.error
import json

url = 'https://eaas-software-prototype-8oj6.onrender.com'

def do_get():
    try:
        r = urllib.request.urlopen(url, timeout=15)
        print(f'GET / -> {r.getcode()}')
        body = r.read(500).decode('utf-8', 'ignore').replace('\n', ' ')
        print(f'Body snippet: {body[:400]}')
    except Exception as e:
        print(f'GET failed: {e!r}')


def do_post():
    login_url = f'{url}/api/login'
    data = json.dumps({}).encode('utf-8')
    req = urllib.request.Request(login_url, data=data, headers={'Content-Type': 'application/json'})
    try:
        r = urllib.request.urlopen(req, timeout=15)
        print(f'POST /api/login -> {r.getcode()}')
        resp = r.read().decode('utf-8', 'ignore')
        print(f'Response: {resp[:1000]}')
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode('utf-8', 'ignore')
        except Exception:
            body = '<no body>'
        print(f'POST HTTPError {e.code} {body}')
    except Exception as e:
        print(f'POST failed: {e!r}')


if __name__ == '__main__':
    do_get()
    do_post()
