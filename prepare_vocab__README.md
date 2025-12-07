# Prepare Vocabulary

To set up the environment and install required packages run

```
prepare_vocab__setup.sh
```

Then to prepare vocabular for language {language_code} run

```
source .venv/bin/activate
python prepare_vocabulary.py --language ${language_code}

```

 I already ran it on the following languages and uploaded the resulting target word files under
 `scdisc_hplt/languages/{language_code}` directory:
 - (in progress) `cmn_Hans`
 - `jpn_Jpan`
 - `mkd_Cyrl`

As of now, the script can run on any language except for these five:
 - `als_Latn` 
 - `bos_Latn` 
 - `kat_Geor` 
 - `tha_Thai`
 - `vie_Latn`

I am working on these and will update the respective directories in a bit. 





