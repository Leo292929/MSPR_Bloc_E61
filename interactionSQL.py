import mysql.connector
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
def result2Info(resultat):

    mydb = mysql.connector.connect(
        host="localhost",
        user="yourusername",
        password="yourpassword",
        database="mydatabase"
    )

    mycursor = mydb.cursor()

    mycursor.execute(f"SELECT * FROM Espece WHERE idEspece = {resultat};")

    for x in mycursor:
        print(x)

    mydb.close()
    


def get_exif_data(image_path,typeData):
    image = Image.open(image_path)
    exif_data = image._getexif()
    if exif_data is not None:
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == typeData:
                return value




def insertIn2Db(path,result,idUser=1):
    try:
        gpsInfo = get_exif_data(path,'GPSInfo')
        gpsInfo = str(gpsInfo)
    except:
        gpsInfo = ""
        print("pas d'info gps")
    try:
        dateTime = get_exif_data(path,'DateTime')
        format_string = '%Y:%m:%d %H:%M:%S'
        dateTime = datetime.strptime(dateTime, format_string)
        date = dateTime.date()
        heure = dateTime.time()
    except:
        date = datetime.now().date()
        heure = datetime.now().time()
        print("pas d'info datetime")



    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="MSPR61"
    )
    mycursor = mydb.cursor()

    sql = "INSERT INTO Empreinte (idUser,idEspece,adresseImage,datePhoto,heurePhoto,localisation) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (idUser, result,path,date,heure,gpsInfo)

    mycursor.execute(sql, val)

    mydb.commit()

    print(mycursor.rowcount, "enregistrement inséré.")

    mydb.close()






