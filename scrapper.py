import os
import os.path as P
import urllib.request as urllib2

from bs4 import BeautifulSoup

def ScrapImages(cat ,links):

    try:
        os.makedirs("Images/%s" % cat)
    except FileExistsError:
        print("Images Folder Already Exists")

    path = P.join(os.getcwd(),"Images",cat)

    print("PATH=",path)

    i=0
    for link in links:
        try:
            soup = BeautifulSoup(urllib2.urlopen(link).read(), features="lxml")
        except:
            print("Netork Error,Check ur Network Connection")
            break
        
        img_links = soup.find_all('img', src=True)
        for img_link in img_links:
            i = i+1    
            img_link = img_link["src"].split("src=")[-1]
            index = img_link.find('?')
            if(index != -1):
                img_link = img_link[:index] #for images to be scrapped bcz they end with ?f=xs so we need to remove that
            print(img_link)
            if(img_link.find("PIAimages") != -1):
                txt = open('%s\%s.jpg' %(path,i) , "wb")
                try:
                    download_img = urllib2.urlopen(img_link)
                except:
                    print("Network Error,Skipping")
                    txt.close()
                    break
                
                txt.write(download_img.read())
                txt.close()


linksBeds =[
    "https://www.ikea.com/in/en/cat/double-beds-16284/",
    "https://www.ikea.com/in/en/cat/single-beds-16285/",
    "https://www.ikea.com/in/en/cat/daybeds-19046/",
    "https://www.ikea.com/in/en/cat/sofa-beds-10663/",
    "https://www.ikea.com/in/en/cat/children-s-beds-8-12-24708/"
]

ScrapImages("beds",linksBeds)

linksChairs =[
    "https://www.ikea.com/in/en/cat/armchairs-chaise-longues-16239/",
    "https://www.ikea.com/in/en/cat/armchairs-chaise-longues-16239/page-2/",
    "https://www.ikea.com/in/en/cat/cafe-chairs-19144/",
    "https://www.ikea.com/in/en/cat/bar-chairs-20864/",
    "https://www.ikea.com/in/en/cat/dining-chairs-25219/",
    "https://www.ikea.com/in/en/cat/step-stools-step-ladders-20611/",
    "https://www.ikea.com/in/en/cat/office-chairs-20652/"

]

ScrapImages("chairs",linksChairs)


linksLighting = [
    "https://www.ikea.com/in/en/cat/floor-lamps-10731/",
    "https://www.ikea.com/in/en/cat/table-lamps-10732/",
    "https://www.ikea.com/in/en/cat/work-lamps-20502/",
    "https://www.ikea.com/in/en/cat/bathroom-lighting-10736/",
    "https://www.ikea.com/in/en/cat/children-s-lighting-18773/",
    "https://www.ikea.com/in/en/cat/wall-lamps-20503/",

]

ScrapImages("lighting",linksLighting)

linksWardrobe = [
    "https://www.ikea.com/in/en/cat/pax-wardrobes-without-doors-19110/",
    "https://www.ikea.com/in/en/cat/pax-wardrobes-with-doors-24337/",
    "https://www.ikea.com/in/en/cat/pax-frames-for-hinge-doors-19106/",
    "https://www.ikea.com/in/en/cat/pax-frames-for-sliding-doors-20087/",
    "https://www.ikea.com/in/en/rooms/free-standing-wardrobes-pub8798ac01"
]

ScrapImages("wardrobe",linksWardrobe)






