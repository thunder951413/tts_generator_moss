import SwiftUI

enum StudioTokens {
    static let space1: CGFloat = 4
    static let space2: CGFloat = 8
    static let space3: CGFloat = 12
    static let space4: CGFloat = 16
    static let space5: CGFloat = 20
    static let space6: CGFloat = 24

    static let innerRadius: CGFloat = 8
    static let elementRadius: CGFloat = 12
    static let containerRadius: CGFloat = 18

    static let compactControlHeight: CGFloat = 34
    static let controlHeight: CGFloat = 40
    static let primaryControlHeight: CGFloat = 48

    static let accentStart = Color(red: 0.40, green: 0.30, blue: 0.98)
    static let accentEnd = Color(red: 0.25, green: 0.52, blue: 0.98)
    static let accent = Color(red: 0.34, green: 0.36, blue: 0.96)
}

struct StudioBackground: View {
    var body: some View {
        ZStack {
            Color(nsColor: .windowBackgroundColor)
            LinearGradient(
                colors: [
                    Color(red: 0.44, green: 0.36, blue: 0.98).opacity(0.10),
                    Color.clear,
                    Color(red: 0.23, green: 0.59, blue: 0.98).opacity(0.08),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        }
        .ignoresSafeArea()
    }
}

struct StudioGlassGroupBoxStyle: GroupBoxStyle {
    func makeBody(configuration: Configuration) -> some View {
        VStack(alignment: .leading, spacing: StudioTokens.space3) {
            configuration.label
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.primary)
            configuration.content
        }
        .padding(StudioTokens.space4)
        .background(
            .regularMaterial,
            in: RoundedRectangle(cornerRadius: StudioTokens.containerRadius, style: .continuous)
        )
        .overlay {
            RoundedRectangle(cornerRadius: StudioTokens.containerRadius, style: .continuous)
                .strokeBorder(.white.opacity(0.22), lineWidth: 0.7)
        }
        .shadow(color: .black.opacity(0.055), radius: 16, y: 7)
    }
}

struct StudioPrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 15, weight: .semibold))
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity, minHeight: StudioTokens.primaryControlHeight)
            .background(
                LinearGradient(
                    colors: [StudioTokens.accentStart, StudioTokens.accentEnd],
                    startPoint: .leading,
                    endPoint: .trailing
                ),
                in: RoundedRectangle(cornerRadius: StudioTokens.elementRadius, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: StudioTokens.elementRadius, style: .continuous)
                    .strokeBorder(.white.opacity(0.30), lineWidth: 0.7)
            }
            .shadow(
                color: StudioTokens.accent.opacity(configuration.isPressed ? 0.12 : 0.28),
                radius: configuration.isPressed ? 4 : 11,
                y: configuration.isPressed ? 2 : 6
            )
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
            .opacity(isEnabled ? (configuration.isPressed ? 0.9 : 1) : 0.45)
    }
}

struct StudioSecondaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    var destructive = false
    var height = StudioTokens.controlHeight

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(destructive ? Color.red : Color.primary)
            .frame(maxWidth: .infinity, minHeight: height)
            .padding(.horizontal, StudioTokens.space3)
            .background(
                .thinMaterial,
                in: RoundedRectangle(cornerRadius: StudioTokens.elementRadius, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: StudioTokens.elementRadius, style: .continuous)
                    .strokeBorder(
                        destructive ? Color.red.opacity(0.25) : Color.primary.opacity(0.12),
                        lineWidth: 0.8
                    )
            }
            .contentShape(RoundedRectangle(cornerRadius: StudioTokens.elementRadius, style: .continuous))
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
            .opacity(isEnabled ? (configuration.isPressed ? 0.75 : 1) : 0.42)
    }
}

struct StudioTintedButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    var destructive = false
    var height = StudioTokens.controlHeight

    func makeBody(configuration: Configuration) -> some View {
        let color = destructive ? Color.red : StudioTokens.accent
        configuration.label
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(color)
            .frame(maxWidth: .infinity, minHeight: height)
            .padding(.horizontal, StudioTokens.space3)
            .background(
                color.opacity(configuration.isPressed ? 0.17 : 0.10),
                in: RoundedRectangle(cornerRadius: StudioTokens.elementRadius, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: StudioTokens.elementRadius, style: .continuous)
                    .strokeBorder(color.opacity(0.24), lineWidth: 0.8)
            }
            .contentShape(RoundedRectangle(cornerRadius: StudioTokens.elementRadius, style: .continuous))
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
            .opacity(isEnabled ? 1 : 0.42)
    }
}

struct StudioToolbarButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .semibold))
            .padding(.horizontal, StudioTokens.space3)
            .frame(height: StudioTokens.compactControlHeight)
            .background(
                .thinMaterial,
                in: RoundedRectangle(cornerRadius: StudioTokens.innerRadius, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: StudioTokens.innerRadius, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.10), lineWidth: 0.7)
            }
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .opacity(isEnabled ? (configuration.isPressed ? 0.75 : 1) : 0.42)
    }
}

struct StudioIconButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .semibold))
            .frame(width: StudioTokens.compactControlHeight, height: StudioTokens.compactControlHeight)
            .background(
                Color.primary.opacity(configuration.isPressed ? 0.10 : 0.055),
                in: Circle()
            )
            .foregroundStyle(.secondary)
            .contentShape(Circle())
    }
}

struct StudioNavigationButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(StudioTokens.space3)
            .background(
                StudioTokens.accent.opacity(configuration.isPressed ? 0.10 : 0.055),
                in: RoundedRectangle(cornerRadius: StudioTokens.elementRadius, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: StudioTokens.elementRadius, style: .continuous)
                    .strokeBorder(StudioTokens.accent.opacity(0.14), lineWidth: 0.8)
            }
            .scaleEffect(configuration.isPressed ? 0.992 : 1)
    }
}

struct StudioStatusPill: View {
    let title: String
    let ready: Bool

    var body: some View {
        HStack(spacing: 7) {
            Circle()
                .fill(ready ? Color.green : Color.orange)
                .frame(width: 8, height: 8)
                .shadow(color: (ready ? Color.green : Color.orange).opacity(0.5), radius: 4)
            Text(title)
                .font(.system(size: 11, weight: .medium))
                .lineLimit(1)
        }
        .padding(.horizontal, 11)
        .frame(height: 30)
        .background(.thinMaterial, in: Capsule())
        .overlay {
            Capsule().strokeBorder(.white.opacity(0.25), lineWidth: 0.6)
        }
    }
}

struct StudioModalHeader: View {
    let title: String
    let subtitle: String
    let close: () -> Void

    var body: some View {
        HStack(spacing: StudioTokens.space4) {
            VStack(alignment: .leading, spacing: StudioTokens.space1) {
                Text(title)
                    .font(.system(size: 20, weight: .semibold, design: .rounded))
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button(action: close) {
                Image(systemName: "xmark")
            }
            .buttonStyle(StudioIconButtonStyle())
            .accessibilityLabel("关闭")
        }
        .padding(.horizontal, StudioTokens.space5)
        .frame(height: 72)
        .background(.ultraThinMaterial)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(.white.opacity(0.20))
                .frame(height: 0.7)
        }
    }
}

struct StudioFooterBar<Content: View>: View {
    @ViewBuilder let content: () -> Content

    var body: some View {
        HStack(spacing: StudioTokens.space2, content: content)
            .padding(.horizontal, StudioTokens.space5)
            .frame(height: 68)
            .background(.ultraThinMaterial)
            .overlay(alignment: .top) {
                Rectangle()
                    .fill(.white.opacity(0.20))
                    .frame(height: 0.7)
            }
    }
}
