#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/13 

from flask import Flask
from flask import redirect

app = Flask(__name__)


@app.route('/')
def index():
    return redirect('/ngrok/site')


@app.route('/ngrok/site')
def ngrok_site():
    update_ngrok_info()

    public_urls = get_ngrok_public_urls()
    if len(public_urls) == 0:
        return '<p> No ngrok public tunnels found, check the debug info: <a href="/ngrok/debug">Ngrok Debug</a> </p>'
    elif len(public_urls) == 1:
        return redirect(public_urls[0])
    else:
        return '<ul>{}</ul>'.format('\n'.join([f'<li><a href="{url}">{url}</a></li>' for url in public_urls]))


@app.route('/ngrok/debug')
def ngrok_debug():
    html = '<div><a href="/ngrok/refresh">Force Refresh State!!</a></div>\n'
    html += format_ngrok_info_html()
    return html


@app.route('/ngrok/refresh')
def ngrok_refresh():
    update_ngrok_info(hayaku=True)
    return redirect('/ngrok/debug')



import os
import time
from copy import deepcopy
from traceback import format_exc
from ngrok.client import Client

API_KEY = None
UPDATE_MIN_INTERVAL = 60

try:
    with open('API_KEY.txt', encoding='utf-8') as fh:
        API_KEY = fh.read().strip()
except: pass
API_KEY = os.environ.get('API_KEY', API_KEY)
ngrok = Client(API_KEY)

ngrok_info = {
    'tunnel_sessions': [ ],
    'tunnels': [ ],
    'endpoints': [ ],
    'update_ts': -1,
    'recent_error': 'API_KEY is null' if API_KEY is None else '',
    'recent_error_ts': time.time() - 2 * UPDATE_MIN_INTERVAL,
}


def get_ngrok_public_urls():
    return [ep['public_url'] for ep in ngrok_info['endpoints']]


def update_ngrok_info(hayaku=False):
    global ngrok_info

    if not hayaku and time.time() - ngrok_info['update_ts'] < UPDATE_MIN_INTERVAL:
        return

    def parse_object(obj):
        r = { }
        for attr in dir(obj):
            if attr.startswith('_'): continue
            try: r[attr] = getattr(obj, attr)
            except: pass
        return r

    try:
        new_ngrok_info = deepcopy(ngrok_info)

        # the local running ngrok.exe client entity
        for ts in ngrok.tunnel_sessions.list():
            new_ngrok_info['tunnel_sessions'].append(parse_object(ts))

        # the tunnels bridging local to public
        for tn in ngrok.tunnels.list():
            new_ngrok_info['tunnels'].append(parse_object(tn))

        # the public service endpoint (like a virtual server)
        for ep in ngrok.endpoints.list():
            new_ngrok_info['endpoints'].append(parse_object(ep))

        new_ngrok_info['update_ts'] = time.time()
        new_ngrok_info['recent_error'] = ''
        new_ngrok_info['recent_error_ts'] = -1
        ngrok_info = new_ngrok_info
    except:
        ngrok_info['recent_error'] = format_exc()
        ngrok_info['recent_error_ts'] = time.time()


def format_ngrok_info_html():
    ts_to_timestr = lambda ts: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts > 0 else 'None'

    mk_div  = lambda text: f'<div>\n{text}\n</div>'
    mk_p    = lambda text: f'<p>{text}</p>'
    mk_span = lambda text: f'<span>{text}</span>'
    mk_h3   = lambda text: f'<h3>{text}</h3>'
    mk_br   = lambda: '<br/>'
    mk_hr   = lambda: '<hr/>'
    mk_line = lambda text: mk_span(text) + mk_br()
    mk_error = lambda text: f'<p style="color:red"><em>{text}</em></p> <br/>'

    stats = ''
    if ngrok_info["recent_error"]:
        stats += mk_error(f'recent_error: {ngrok_info["recent_error"]}')
        stats += mk_error(f'recent_error_ts: {ts_to_timestr(ngrok_info["recent_error_ts"])}')
        stats + mk_hr()
    stats += mk_line(f'update_ts: {ts_to_timestr(ngrok_info["update_ts"])}')

    eps = mk_h3('[Endpoints]')
    for ep in ngrok_info['endpoints']:
        eps += mk_p('\n'.join([mk_line(f'{k}: {v}') for k, v in ep.items()]))
    
    tns = mk_h3('[Tunnels]')
    for tn in ngrok_info['tunnels']:
        tns += mk_p('\n'.join([mk_line(f'{k}: {v}') for k, v in tn.items()]))

    tss = mk_h3('[Tunnel Sessions]')
    for ts in ngrok_info['tunnel_sessions']:
        tss += mk_p('\n'.join([mk_line(f'{k}: {v}') for k, v in ts.items()]))
    
    return '\n'.join([mk_div(stats), mk_hr(), mk_div(eps), mk_hr(), mk_div(tns), mk_hr(), mk_div(tss)])
