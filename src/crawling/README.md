# Devpost gallery crawling

The `crawling/` package uses Selenium to discover projects and GitHub links from a Devpost gallery. It does not grade repositories or call Ollama; those steps belong to the [HackGrader workflow](../hackgrader/README.md).

## Package map

| File | Responsibility |
| --- | --- |
| `__init__.py` | Exposes `Crawler` and `Driver`. |
| `driver.py` | Creates and terminates the Selenium browser. |
| `crawler.py` | Opens gallery and project pages, then collects project and GitHub URLs. |

## Runtime behavior

Gallery crawling requires network access and a working Chrome installation. The application coordinator owns the browser lifecycle and terminates the driver even if crawling raises an error. The browser is closed before model readiness checks and repository grading begin.

Devpost protects its pages with a bot check that refuses the `HeadlessChrome` user agent, and then answers every page load after the first one in a browser session with a verification page it never clears. The driver presents the installed browser's own user agent and clears cookies before each navigation, so gallery pagination and project pages load normally. A page that is still behind the check is reported as a bot check rather than as changed markup.

The crawler returns discovered links; the workflow filters and deduplicates GitHub repository slugs before grading. A gallery without any usable GitHub projects stops rather than producing an empty report. Browser and page changes can affect live results, so tests that check application logic should mock `Crawler` and `Driver` rather than rely on a live gallery.

## Testing

Run deterministic tests with the [offline test commands](../../tests/README.md). The live crawling test contacts Devpost and needs Chrome and network access:

```bash
bash run_test.sh live
```

On PowerShell, use `./run_test.ps1 -Mode live`. Live tests may depend on the current state of the configured example gallery.
