"""Create/push this repository and enable Pages, using the user's normal gh login."""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OWNER, REPO = 'Rohan-Ktksu', 'ercot-peak-demand-lab'
FULL = OWNER + '/' + REPO


def run(args, check=True):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if check and result.returncode:
        raise SystemExit(result.stderr.strip() or result.stdout.strip() or 'Command failed.')
    return result


login = run(['gh', 'api', 'user', '--jq', '.login']).stdout.strip()
if login.casefold() != OWNER.casefold():
    raise SystemExit('The active GitHub account is not Rohan-Ktksu. Stop and select the intended account.')
metadata = run(['gh', 'api', 'repos/' + FULL], check=False)
if metadata.returncode:
    if '404' not in metadata.stderr:
        raise SystemExit(metadata.stderr.strip())
    created = run(['gh', 'repo', 'create', FULL, '--public', '--description',
        'Rohan Sai’s ERCOT peak-demand research: 24-feature ML model, three-hour windows, reproducible notebook and interactive website.'])
    print('Created repository:', created.stdout.strip(), flush=True)
else:
    info = json.loads(metadata.stdout)
    if info.get('owner', {}).get('login', '').casefold() != OWNER.casefold():
        raise SystemExit('Repository owner differs from the intended user.')
    print('Using existing repository:', info['html_url'], flush=True)
url = f'https://github.com/{FULL}.git'
remote = run(['git', 'remote', 'get-url', 'origin'], check=False)
if remote.returncode:
    run(['git', 'remote', 'add', 'origin', url])
elif remote.stdout.strip() != url:
    raise SystemExit('Existing origin differs; preserve it and review before pushing.')
run(['gh', 'auth', 'setup-git'])
pushed = run(['git', 'push', '--set-upstream', 'origin', 'main'])
print(pushed.stdout.strip() or pushed.stderr.strip(), flush=True)
endpoint = 'repos/' + FULL + '/pages'
pages = run(['gh', 'api', endpoint], check=False)
if pages.returncode:
    if '404' not in pages.stderr:
        raise SystemExit(pages.stderr.strip())
    pages = run(['gh', 'api', '--method', 'POST', endpoint,
        '-f', 'source[branch]=main', '-f', 'source[path]=/docs', '-f', 'build_type=legacy'])
info = json.loads(pages.stdout)
if info.get('source', {}).get('branch') != 'main' or info.get('source', {}).get('path') != '/docs':
    raise SystemExit('Pages exists with another source. Review it rather than overwriting its configuration.')
receipt = dict(repository=f'https://github.com/{FULL}', website=info['html_url'],
    source=info['source'], status=info.get('status'), commit=run(['git', 'rev-parse', 'HEAD']).stdout.strip())
(ROOT.parent / 'ercot-github-deployment.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2), flush=True)
