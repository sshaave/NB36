# NB36
Kode for karbonfiberforsterkede betongkonstruksjoner

I dette repoet ligger et beregningsprogram for fritt opplagt betongbjelke med jevnt fordelt last.
Programmet er laget av Simen Haave fra Novelsolutions.no i forbindelse med NB36.

## Omfang
Dette programmet regner på fritt opplagte bjelker med jevnt fordelt last. Betongbjelkens tverrsnitt kan være av alle typer,
og må defineres ved bruk av "width_function" rett over main-funksjonen. Hvis bjelken har varierende tverrsnittshøyde (SDT),
må height være en vektor for som har samme størrelse som momentvektor.

Materialer:
    - betong med og uten strekkfasthet (strekkfasthet kun i SLS)
    - vanlig slakkarmering B500NC eller B400NC
    - spennarmering (pass på spennkraft, programmet setter den til 0 aka forblender den, hvis likevekt ikke oppnås i gitt snitt)
    - karbonfiber

På grunn av programmet er satt opp for å ha strekksone kun i UK kan ikke kontinuerlige bjelker regnes på.

For andre typer laster (punktlaster) må brukeren selv sette opp momentvektorene.

Svakheter og begrensninger: 
    - Takler ikke strekk i OK -> vil aldri få pilhøyde
    - Føroppspenning: input er aktiv kraft, tap av spennkraft må bruker ta hensyn til
    - Føroppspenning: Delvis forblending er løst ved å senke beregningsmessig spennkraft i aktuelt snitt
    - Kryp- og svinnverdier må brukeren gi
    - Skjærkapasitet ikke kontrollert
    - Kun jevnt fordelte laster lagt inn


## Installasjon
Koden er skrevet i Python og krever en lokal installasjon av python
Klon repoet, og bruk filen beam.py for å regne på bjelker. Se på eksempel_1.py for inspirasjon