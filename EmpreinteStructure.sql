drop database if exists MSPR61;
create database MSPR61;
use MSPR61;

Drop table if exists Empreintes;
Drop table if exists User;

CREATE TABLE IF NOT EXISTS Empreintes(
 idEmpreinte int NOT NULL AUTO_INCREMENT,
 idUser int NOT NULL,
 espece ENUM('castor','chat','chien','coyote','ecureuil','lapin','loup','lynx','ours','puma','rat','raton laveur','renard'),
 adresseImage varchar(255),
 datePhoto date,
 heurePhoto time,
 localisation varchar(20),
 PRIMARY KEY(idEmpreinte),
 FOREIGN KEY (idUser) REFERENCES User(idUser)
 );
 
 CREATE TABLE  IF NOT EXISTS User(
 idUser int NOT NULL AUTO_INCREMENT,
 courielUser varchar(255),
 mdpUser varchar(255),
 PRIMARY KEY(idUser)
 );
 