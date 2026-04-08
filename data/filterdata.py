import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

def remove_www_prefix(domains):
    result = []
    for domain in domains:
        d = str(domain).strip().lower()

        if not d.startswith(('http://', 'https://')):
            d = 'http://' + d
        parsed = urlparse(d)
        hostname = parsed.hostname or ''       
        hostname = re.sub(r'^www\.', '', hostname)
        result.append(hostname)
    return np.array(result)


phi = pd.read_csv("PhiUSIIL/PhiUSIIL.csv", index_col=False)
phi['Domain'] = remove_www_prefix(phi['Domain'].values)
phi.to_csv("PhiUSIIL/phiusiil-filtered.csv", index=False)

bcp = pd.read_csv("less-is-more/BTCP.csv", index_col=False)
bcp['0'] = remove_www_prefix(bcp['0'].values)
bcp.to_csv("less-is-more/BTCP.csv", index=False)


