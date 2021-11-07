from json import dumps
import os
import requests

keyword = 'dog'
accesskey = 'bAXayUDgC0EKt_Qtkv-JUH2GrBoudftEWk_Z5nTXvmg'
folder = './data/buddydog/'
# 
endpoint = 'https://api.unsplash.com/'
quantity = 100
perpage = 30


def main():
    page = 0
    current = 0
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    while current < quantity:
        page += 1
        needed = quantity - current > perpage and perpage or quantity - current
        data = requests.get(endpoint + 'search/photos',
                            params={'query': keyword, 'orientation': 'landscape', 'per_page': needed, 'page': page}, headers={'Authorization': 'Client-ID '+accesskey}).json()
        for index, result in enumerate(data['results']):
            t = downloadImage(result['urls']['small'])
            if t.status_code == 200:
                dumpsFile(folder + str(current+index) + '.jpg', t.content)
                printProgressBar(current+index, quantity,
                                 prefix='Progress:', suffix='Complete ')
            else:
                print(t.content)
        current += perpage

def downloadImage(url):
    return requests.get(url, stream=True)

def dumpsFile(path, data):
    with open(path, 'wb+') as f:
        f.write(data)


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == "__main__":
    main()
