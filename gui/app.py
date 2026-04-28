import dash
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="AutoEIS GUI",
)

STEPS = [
    ("① Data", "/"),
    ("② Mode", "/mode"),
    ("③ Fitting", "/fitting"),
    ("④ Results", "/results"),
]

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand(
                html.Span(
                    [html.I(className="fa fa-bolt me-2 text-warning"), "AutoEIS GUI"],
                    className="fw-bold fs-5",
                )
            ),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink(label, href=href, active="exact", className="fw-semibold"))
                    for label, href in STEPS
                ],
                className="ms-auto gap-1",
                navbar=True,
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
    sticky="top",
    className="mb-4 shadow",
)

app.layout = html.Div(
    [
        # Shared state stores — all in app layout so they survive page navigation
        dcc.Store(id="store-data", storage_type="memory"),
        dcc.Store(id="store-mode", storage_type="memory"),
        dcc.Store(id="store-task-id", storage_type="memory"),
        dcc.Store(id="store-results", storage_type="memory"),
        # Data-page stores (moved here so data page state survives navigation)
        dcc.Store(id="store-raw", storage_type="memory"),
        dcc.Store(id="store-deleted", data=[]),
        dcc.Store(id="store-pending-delete", data=[]),
        dcc.Store(id="store-df-cols", storage_type="memory"),
        navbar,
        dbc.Container(dash.page_container, fluid=True, className="pb-5"),
    ]
)

server = app.server  # expose for gunicorn/deployment

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
