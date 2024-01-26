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
