drop database if exists MSPR61;
create database MSPR61;
use MSPR61;

Drop table if exists Empreinte;
Drop table if exists User;

CREATE TABLE IF NOT EXISTS Empreinte(
 idEmpreinte int NOT NULL AUTO_INCREMENT,
 idUser int NOT NULL,
 idEspece int NOT NULL,
 adresseImage varchar(255),
 datePhoto date,
 heurePhoto time,
 localisation varchar(20),
 PRIMARY KEY(idEmpreinte),
 FOREIGN KEY (idUser) REFERENCES User(idUser),
 FOREIGN KEY (idEspece) REFERENCES Espece(idEspece)
 );
 
CREATE TABLE  IF NOT EXISTS Utilisateur(
 idUser int NOT NULL AUTO_INCREMENT,
 nomUser varchar(255),
 courielUser varchar(255),
 mdpUser varchar(255),
 PRIMARY KEY(idUser)
 );
 
CREATE TABLE  IF NOT EXISTS Espece(
 idEspece int NOT NULL AUTO_INCREMENT,
 nomEspece varchar(255),
 descriptionEspece varchar(255),
 nomLatin varchar(255),
 famille ENUM('Mamifere'),
 taille varchar(255),
 region varchar(255),
 habitat varchar(255),
 funfact TEXT,
 PRIMARY KEY(idEspece)
 );


INSERT INTO Espece (nomEspece, descriptionEspece, nomLatin, famille, taille, region, habitat, funfact) 
VALUES ('Castor', 'Le castor d’Europe, appelé également le castor commun ou le castor d’Eurasie, est un mammifère rongeur aquatique de la famille des castoridés.', 'Castor canadensis', 'Mamifere', '100 à 135 cm queue comprise', 'Europe du nord et Eurasie', 'Le castor d’Europe vit le long des rivières, des ruisseaux, des lacs et des étangs.', 'À l’exception des humains, le castor est l’un des seuls mammifères qui façonnent son environnement.');

INSERT INTO Utilisateur ( nomUser,courielUser,mdpUser)
VALUES ('root','root@root.com','root');