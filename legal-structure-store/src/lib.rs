use legal_structure::SourceDoc;
use rusqlite::{params, Connection, OpenFlags, OptionalExtension, Transaction};
use std::path::{Path, PathBuf};

pub struct CachedSourceDoc {
    pub document: Vec<u8>,
    pub index: Vec<u8>,
}

pub struct SourceDocStore {
    connection: Connection,
}

impl SourceDocStore {
    pub fn open(path: impl AsRef<Path>) -> rusqlite::Result<Self> {
        let connection = Connection::open(path)?;
        connection.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS source_doc (
               provider TEXT NOT NULL,
               source_id TEXT NOT NULL,
               document_id TEXT NOT NULL,
               revision TEXT NOT NULL,
               document BLOB NOT NULL,
               index_entries BLOB NOT NULL,
               PRIMARY KEY (provider, source_id)
             ) WITHOUT ROWID;
             CREATE INDEX IF NOT EXISTS source_doc_document
               ON source_doc(provider, document_id);
             CREATE TABLE IF NOT EXISTS source_doc_meta (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
             ) WITHOUT ROWID;",
        )?;
        Ok(Self { connection })
    }

    pub fn open_read_only(path: impl AsRef<Path>) -> rusqlite::Result<Self> {
        Ok(Self {
            connection: Connection::open_with_flags(
                path,
                OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )?,
        })
    }

    pub fn write<'a>(&'a mut self) -> rusqlite::Result<SourceDocWriter<'a>> {
        let transaction = self.connection.transaction()?;
        Ok(SourceDocWriter { transaction })
    }

    pub fn meta(&self, key: &str) -> rusqlite::Result<Option<String>> {
        self.connection
            .query_row(
                "SELECT value FROM source_doc_meta WHERE key=?1",
                [key],
                |row| row.get(0),
            )
            .optional()
    }

    pub fn clear(&mut self) -> rusqlite::Result<()> {
        self.connection
            .execute_batch("DELETE FROM source_doc; DELETE FROM source_doc_meta;")
    }

    pub fn len(&self) -> rusqlite::Result<usize> {
        let count: i64 =
            self.connection
                .query_row("SELECT COUNT(*) FROM source_doc", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    pub fn finish(&self) -> rusqlite::Result<()> {
        self.connection.execute_batch(
            "PRAGMA wal_checkpoint(TRUNCATE);
             PRAGMA journal_mode=DELETE;
             PRAGMA optimize;",
        )
    }

    pub fn get(
        &self,
        provider: &str,
        source_id: &str,
    ) -> rusqlite::Result<Option<CachedSourceDoc>> {
        self.connection
            .query_row(
                "SELECT document,index_entries FROM source_doc
             WHERE provider=?1 AND source_id=?2",
                params![provider, source_id],
                |row| {
                    Ok(CachedSourceDoc {
                        document: row.get(0)?,
                        index: row.get(1)?,
                    })
                },
            )
            .optional()
    }
}

fn partial_path(target: &Path) -> PathBuf {
    let mut value = target.as_os_str().to_owned();
    value.push(".partial");
    value.into()
}

#[cfg(not(windows))]
fn replace_file(source: &Path, target: &Path) -> std::io::Result<()> {
    std::fs::rename(source, target)
}

#[cfg(windows)]
fn replace_file(source: &Path, target: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
    #[link(name = "kernel32")]
    extern "system" {
        fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    }
    let source = source
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let target = target
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    if unsafe {
        MoveFileExW(
            source.as_ptr(),
            target.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    } == 0
    {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

pub struct SourceDocWriter<'a> {
    transaction: Transaction<'a>,
}

impl SourceDocWriter<'_> {
    pub fn meta(&self, key: &str, value: &str) -> rusqlite::Result<()> {
        self.transaction.execute(
            "INSERT INTO source_doc_meta(key,value) VALUES(?1,?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn put(
        &self,
        source_id: &str,
        document: &SourceDoc,
        include_text: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let provider = document
            .provider
            .ok_or("SourceDoc provider is required")?
            .as_str();
        self.transaction.execute(
            "INSERT INTO source_doc(provider,source_id,document_id,revision,document,index_entries)
             VALUES(?1,?2,?3,?4,?5,?6)
             ON CONFLICT(provider,source_id) DO UPDATE SET
               document_id=excluded.document_id,
               revision=excluded.revision,
               document=excluded.document,
               index_entries=excluded.index_entries",
            params![
                provider,
                source_id,
                document.id,
                document.revision,
                document.json_bytes(include_text)?,
                serde_json::to_vec(&document.index.entries())?,
            ],
        )?;
        Ok(())
    }

    pub fn commit(self) -> rusqlite::Result<()> {
        self.transaction.commit()
    }
}

#[cfg(feature = "a2aj")]
pub mod a2aj {
    use super::{partial_path, replace_file, SourceDocStore};
    use legal_structure::{a2aj_source_doc, A2ajInput, A2ajSourceKind};
    use std::path::Path;

    const BATCH_SIZE: usize = 10_000;

    fn text(row: &rusqlite::Row<'_>, name: &str) -> rusqlite::Result<Option<String>> {
        row.get::<_, Option<String>>(name)
    }

    fn field(
        row: &rusqlite::Row<'_>,
        name: &str,
        language: &str,
    ) -> rusqlite::Result<Option<String>> {
        let other = if language == "en" { "fr" } else { "en" };
        Ok(text(row, &format!("{name}_{language}"))?
            .filter(|value| !value.trim().is_empty())
            .or(text(row, &format!("{name}_{other}"))?.filter(|value| !value.trim().is_empty())))
    }

    fn array_index(key: &str) -> Option<u32> {
        let value = key.parse::<u32>().ok()?;
        (value != u32::MAX && value.to_string() == key).then_some(value)
    }

    fn section_map(
        value: Option<String>,
    ) -> Result<Option<Vec<(String, String)>>, Box<dyn std::error::Error>> {
        let Some(value) = value else { return Ok(None) };
        let parsed: serde_json::Value = serde_json::from_str(&value)?;
        let object = parsed.as_object().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "A2AJ section map must be a JSON object",
            )
        })?;
        let mut indexed = Vec::new();
        let mut named = Vec::new();
        for (key, value) in object {
            let Some(value) = value.as_str() else {
                continue;
            };
            if let Some(index) = array_index(key) {
                indexed.push((index, key.clone(), value.to_owned()));
            } else {
                named.push((key.clone(), value.to_owned()));
            }
        }
        indexed.sort_by_key(|entry| entry.0);
        let mut entries = Vec::with_capacity(indexed.len() + named.len());
        entries.extend(indexed.into_iter().map(|(_, key, value)| (key, value)));
        entries.extend(named);
        Ok((!entries.is_empty()).then_some(entries))
    }

    #[derive(Debug, Eq, PartialEq)]
    pub struct ImportSummary {
        pub processed: usize,
        pub total: usize,
        pub complete: bool,
    }

    fn matches(
        store: &SourceDocStore,
        source_size: &str,
        source_mtime_ms: &str,
        engine_version: &str,
        complete: Option<&str>,
    ) -> bool {
        let meta = |key| store.meta(key).ok().flatten();
        meta("store_schema").as_deref() == Some("1")
            && meta("provider").as_deref() == Some("a2aj")
            && meta("engine_version").as_deref() == Some(engine_version)
            && meta("source_size").as_deref() == Some(source_size)
            && meta("source_mtime_ms").as_deref() == Some(source_mtime_ms)
            && complete.is_none_or(|value| meta("complete").as_deref() == Some(value))
    }

    pub fn import(
        source: impl AsRef<Path>,
        target: impl AsRef<Path>,
        row_limit: Option<usize>,
        mut progress: impl FnMut(usize),
    ) -> Result<ImportSummary, Box<dyn std::error::Error>> {
        let source_path = source.as_ref();
        let target = target.as_ref();
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let metadata = source.as_ref().metadata()?;
        let source = rusqlite::Connection::open_with_flags(
            source_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        let source_size = metadata.len().to_string();
        let source_mtime_ms = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis()
            .to_string();
        let engine_version = legal_structure::SOURCE_DOC_VERSION.to_string();
        if target.exists() {
            if let Ok(store) = SourceDocStore::open_read_only(target) {
                if matches(
                    &store,
                    &source_size,
                    &source_mtime_ms,
                    &engine_version,
                    Some("1"),
                ) {
                    return Ok(ImportSummary {
                        processed: 0,
                        total: store.len()?,
                        complete: true,
                    });
                }
            }
        }
        let partial = partial_path(target);
        let mut store = SourceDocStore::open(&partial)?;
        let reusable = matches(
            &store,
            &source_size,
            &source_mtime_ms,
            &engine_version,
            None,
        );
        if !reusable {
            store.clear()?;
        }
        let resume_id = if reusable {
            store
                .meta("last_source_row")?
                .and_then(|value| value.parse::<i64>().ok())
                .unwrap_or(0)
        } else {
            0
        };
        let mut query = source.prepare(
            "SELECT id,doc_type,dataset,citation_en,citation_fr,citation2_en,citation2_fr,
                    name_en,name_fr,url_en,url_fr,unofficial_text_en,unofficial_text_fr,
                    unofficial_sections_en,unofficial_sections_fr
             FROM document WHERE id > ?1 ORDER BY id",
        )?;
        let mut rows = query.query([resume_id])?;
        let previous = if reusable { store.len()? } else { 0 };
        let mut writer = store.write()?;
        writer.meta("store_schema", "1")?;
        writer.meta("provider", "a2aj")?;
        writer.meta("engine_version", &engine_version)?;
        writer.meta("source_size", &source_size)?;
        writer.meta("source_mtime_ms", &source_mtime_ms)?;
        writer.meta("complete", "0")?;
        let mut count = previous;
        let mut processed = 0;
        let mut processed_rows = 0;
        let mut committed_rows = 0;
        let mut last_complete_row = resume_id;
        let mut exhausted = false;
        while row_limit.is_none_or(|limit| processed_rows < limit) {
            let Some(row) = rows.next()? else {
                exhausted = true;
                break;
            };
            let row_id = row.get::<_, i64>("id")?;
            let source_kind = if text(row, "doc_type")?.as_deref() == Some("laws") {
                A2ajSourceKind::Laws
            } else {
                A2ajSourceKind::Cases
            };
            for language in ["en", "fr"] {
                let Some(value) = text(row, &format!("unofficial_text_{language}"))?
                    .filter(|value| !value.trim().is_empty())
                else {
                    continue;
                };
                let Some(citation) =
                    field(row, "citation", language)?.or(field(row, "citation2", language)?)
                else {
                    continue;
                };
                let mut input = A2ajInput::new(citation, source_kind, value);
                input.dataset = text(row, "dataset")?;
                input.name = field(row, "name", language)?;
                input.url = field(row, "url", language)?;
                input.alternate_citation = field(row, "citation2", language)?;
                input.section_map = section_map(field(row, "unofficial_sections", language)?)?;
                writer.put(
                    &format!("{row_id}:{language}"),
                    &a2aj_source_doc(input)?,
                    false,
                )?;
                count += 1;
                processed += 1;
            }
            processed_rows += 1;
            last_complete_row = row_id;
            if processed_rows - committed_rows >= BATCH_SIZE {
                writer.meta("last_source_row", &last_complete_row.to_string())?;
                writer.meta("document_count", &count.to_string())?;
                writer.commit()?;
                committed_rows = processed_rows;
                progress(count);
                writer = store.write()?;
            }
        }
        if !exhausted && row_limit.is_some_and(|limit| processed_rows == limit) {
            exhausted = rows.next()?.is_none();
        }
        writer.meta("last_source_row", &last_complete_row.to_string())?;
        writer.meta("document_count", &count.to_string())?;
        writer.meta("complete", if exhausted { "1" } else { "0" })?;
        writer.commit()?;
        if processed_rows != committed_rows {
            progress(count);
        }
        if exhausted {
            drop(rows);
            drop(query);
            drop(source);
            store.finish()?;
            drop(store);
            replace_file(&partial, target)?;
        }
        Ok(ImportSummary {
            processed,
            total: count,
            complete: exhausted,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::time::{SystemTime, UNIX_EPOCH};

        #[test]
        fn resumes_rows_and_only_promotes_a_complete_store(
        ) -> Result<(), Box<dyn std::error::Error>> {
            let root = std::env::temp_dir().join(format!(
                "legal-structure-store-{}-{}",
                std::process::id(),
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
            ));
            std::fs::create_dir_all(&root)?;
            let source = root.join("a2aj.sqlite");
            let target = root.join("sourcedocs.sqlite");
            let database = rusqlite::Connection::open(&source)?;
            database.execute_batch(
                "CREATE TABLE document (
                   id INTEGER PRIMARY KEY, doc_type TEXT, dataset TEXT,
                   citation_en TEXT, citation_fr TEXT, citation2_en TEXT, citation2_fr TEXT,
                   name_en TEXT, name_fr TEXT, url_en TEXT, url_fr TEXT,
                   unofficial_text_en TEXT, unofficial_text_fr TEXT,
                   unofficial_sections_en TEXT, unofficial_sections_fr TEXT
                 );
                 INSERT INTO document VALUES (
                   1,'cases','ONCA','2026 ONCA 1',NULL,NULL,NULL,
                   'First case',NULL,NULL,NULL,'1 First paragraph.',NULL,NULL,NULL
                 );
                 INSERT INTO document VALUES (
                   2,'laws','ON','Test Act','Loi test',NULL,NULL,
                   'Test Act','Loi test',NULL,NULL,
                   '34(2) Parent provision.\n(a) Child provision.',
                   '34(2) Disposition principale.\n(a) Disposition enfant.',
                   '{\"34\":\"34(2) Parent provision.\\n(a) Child provision.\"}',
                   '{\"34\":\"34(2) Disposition principale.\\n(a) Disposition enfant.\"}'
                 );",
            )?;
            drop(database);

            let first = import(&source, &target, Some(1), |_| {})?;
            assert_eq!(
                first,
                ImportSummary {
                    processed: 1,
                    total: 1,
                    complete: false,
                }
            );
            assert!(!target.exists());
            assert!(partial_path(&target).exists());

            let second = import(&source, &target, Some(1), |_| {})?;
            assert_eq!(
                second,
                ImportSummary {
                    processed: 2,
                    total: 3,
                    complete: true,
                }
            );
            assert!(target.exists());
            assert!(!partial_path(&target).exists());
            let store = SourceDocStore::open_read_only(&target)?;
            assert_eq!(store.meta("complete")?.as_deref(), Some("1"));
            let cached = store.get("a2aj", "2:en")?.expect("English law SourceDoc");
            let document: serde_json::Value = serde_json::from_slice(&cached.document)?;
            assert!(document["blocks"]
                .as_array()
                .unwrap()
                .iter()
                .any(|block| block["label"] == "sec34(2)(a)"));
            drop(store);

            assert_eq!(
                import(&source, &target, None, |_| {})?,
                ImportSummary {
                    processed: 0,
                    total: 3,
                    complete: true,
                }
            );
            std::fs::remove_dir_all(root)?;
            Ok(())
        }
    }
}
