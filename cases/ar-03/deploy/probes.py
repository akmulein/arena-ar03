"""Existing live researcher isolation probe; executed with isolated Python."""
# ruff: noqa: E501

RESEARCHER = r"""
import errno,json,os,socket,sqlite3,subprocess,sys
from pathlib import Path
import httpx

def reachable(host,port):
    try:
        with socket.create_connection((host,port),timeout=.5):
            return True
    except OSError:
        return False

def journal_readonly():
    with sqlite3.connect('/search/search.sqlite3') as database:
        database.execute('SELECT max_solutions FROM campaign').fetchone()
        try:
            database.execute('BEGIN IMMEDIATE')
            database.execute('UPDATE campaign SET max_solutions=max_solutions')
        except sqlite3.OperationalError as error:
            database.rollback()
            return 'readonly' in str(error).lower() or 'read-only' in str(error).lower()
        database.rollback()
    return False

def train_readonly():
    try:
        descriptor=os.open('/workspace/cases/ar-03/_local/train-500.jsonl',os.O_WRONLY|os.O_APPEND)
    except OSError as error:
        return error.errno in (errno.EROFS,errno.EACCES,errno.EPERM)
    os.close(descriptor)
    return False

def dns_blocked():
    query=b'\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01'
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as connection:
        connection.settimeout(1)
        try:
            connection.sendto(query,('127.0.0.11',53))
            connection.recvfrom(512)
            return False
        except OSError:
            return True

def proxy_blocked():
    try:
        with httpx.Client(proxy=os.environ['HTTPS_PROXY'],trust_env=False,timeout=3) as client:
            client.get('https://example.com')
    except httpx.ProxyError as error:
        return '403' in str(error)
    return False

host_paths=json.loads(sys.argv[1])
token=Path('/run/secrets/submit-token').read_text().strip()
with httpx.Client(trust_env=False,timeout=3) as client:
    permitted=client.get(os.environ['AR03_API']+'/status',headers={'Authorization':'Bearer '+token})
    denied=client.get(os.environ['AR03_API']+'/status')
    admin=client.post(os.environ['AR03_API']+'/admin/stop',json={},headers={'Authorization':'Bearer '+token})
status=permitted.json()
checks={
 'unprivileged':os.getuid()==10001,
 'no_capabilities':'CapEff:\t0000000000000000' in Path('/proc/self/status').read_text(),
 'operator_files_and_tokens_hidden':all(not Path(path).exists() and not os.access(path,os.R_OK) for path in host_paths),
 'docker_socket_hidden':all(not Path(path).exists() for path in ('/var/run/docker.sock','/run/docker.sock')),
 'operator_env_hidden':not any(os.environ.get(key) for key in ('AR03_ADMIN_TOKEN','AR03_TRAIN_TOKEN','AR03_PUBLIC_TOKEN','AR03_PRIVATE_TOKEN','ARENA_ADMIN_TOKEN','DOCKER_HOST','DOCKER_CERT_PATH')),
 'cannot_change_network':subprocess.run(['iptables','-S'],capture_output=True).returncode!=0,
 'journal_readonly':journal_readonly(),
 'train_readonly':train_readonly(),
 'direct_internet_blocked':not reachable('1.1.1.1',443),
 'direct_host_api_blocked':not reachable('192.168.65.254',8799),
 'direct_train_api_blocked':not reachable('192.168.65.254',8800),
 'docker_tcp_blocked':not any(reachable('192.168.65.254',port) for port in (2375,2376)),
 'embedded_dns_blocked':dns_blocked(),
 'unapproved_proxy_target_blocked':proxy_blocked(),
 'authorized_api_only':permitted.status_code==200 and denied.status_code==401,
 'operator_admin_api_blocked':admin.status_code==404,
 'evolution_locked':not status['authorized'],
}
print(json.dumps(checks))
"""
